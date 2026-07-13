---
weight: 4800
title: "Chapter 21"
description: "Computational Geometry"
icon: "article"
date: "2026-07-06T00:00:00+07:00"
lastmod: "2026-07-06T00:00:00+07:00"
katex: true
draft: false
toc: true
---

{{% alert icon="💡" context="info" %}}
<strong><em>“Geometry gives form to the universe; computational geometry gives motion to that form, transforming abstract space into something the machine can reason about.”</em></strong>
{{% /alert %}}

{{% alert icon="📘" context="success" %}}
<p style="text-align: justify;"><em>Chapter 21 introduces the foundations of computational geometry and their applications in scientific computing. The chapter begins with geometric representations based on points, simplices, meshes, and point clouds, together with the predicates used to make geometric decisions. Fundamental concepts such as orientation, incircle and insphere tests, barycentric coordinates, and point-classification methods are developed as building blocks for robust geometric computation. The chapter then examines algorithms for intersections, clipping, convex hull construction, Delaunay triangulations, and Voronoi diagrams. Spatial-search techniques, including grids, tree-based structures, and nearest-neighbor methods, are presented alongside mesh-quality analysis, interpolation, and point-location algorithms. Particular emphasis is placed on robustness in finite-precision arithmetic, including degeneracy handling, exact predicates, filtering techniques, and reliable geometric pipelines. Throughout the chapter, mathematical concepts are integrated with practical Rust implementations, providing readers with the tools needed to develop accurate, efficient, and robust geometric algorithms for modern scientific computing applications.</em></p>
{{% /alert %}}

# 21.1. Introduction

Computational geometry provides the discrete geometric infrastructure on which many numerical methods depend. In scientific computing, geometry is not merely a device for visualization or graphical rendering. It determines the algebraic and topological structure on which finite element, finite volume, meshless, particle, remapping, and point-cloud algorithms operate. A numerical method may ultimately solve a system of equations, approximate an integral, evolve a differential equation, or minimize a functional, but before any of these operations can be performed reliably, the computational domain must be represented, partitioned, queried, and tested. Thus, the geometric layer supplies the points, vectors, cells, faces, intersections, neighborhoods, coordinate maps, and adjacency relations that make the later numerical calculation meaningful.

The objects of interest in this chapter are the geometric objects that scientific codes actually manipulate: points in $\mathbb{R}^d$, vectors, segments, polygons, polyhedra, simplices, structured and unstructured meshes, point clouds, and spatial data structures. These objects enter numerical computation through two complementary roles. First, they provide a representation of the physical or computational domain. Second, they support decision procedures, usually called geometric predicates, that determine orientation, sidedness, containment, intersection, adjacency, proximity, and local coordinate membership. Modern work on robust predicates, exact mesh arrangements, constructive solid geometry, Delaunay and Voronoi computation, point-cloud search, and finite element mesh generation emphasizes that these geometric decisions are not secondary bookkeeping operations. They are part of the numerical correctness of the complete simulation pipeline (Bartels, Fisikopoulos and Weiser, 2023; Guo and Fu, 2024; Lévy, 2025; Elshakhs et al., 2024; Lei et al., 2024).

## 21.1.1. Geometry as Representation and Decision

A point in $d$-dimensional Euclidean space is written as:

$$p = (p_1,p_2,\ldots,p_d) \in \mathbb{R}^d \tag{21.1.1}$$

A vector is the difference between two points,

$$v = q-p,\qquad v_i = q_i-p_i,\quad i=1,\ldots,d,\tag{21.1.2}$$

and the Euclidean metric is induced by the inner product:

$$u \cdot v = \sum_{i=1}^d u_i v_i,\qquad \|v\|_2 = \sqrt{v\cdot v} \tag{21.1.3}$$

These elementary definitions become algorithmically significant as soon as geometric tests are used as branch conditions. For example, the decision that a point lies to the left or right of an oriented segment in the plane is determined by the sign of a determinant:

$$
\operatorname{orient2d}(a,b,c)
=
\det
\begin{bmatrix}
a_x-c_x & a_y-c_y\\
b_x-c_x & b_y-c_y
\end{bmatrix}
\tag{21.1.4}
$$

\
Equivalently,

$$
\operatorname{orient2d}(a,b,c)
=
(a_x-c_x)(b_y-c_y)
-
(a_y-c_y)(b_x-c_x)
\tag{21.1.5}
$$

If this quantity is positive, the ordered triple $(a,b,c)$ has one orientation; if it is negative, it has the opposite orientation; if it is zero, the three points are collinear. This single predicate underlies segment intersection, polygon winding, convex hull construction, Delaunay edge flips, mesh orientation checks, and many point-location algorithms.

In three dimensions, the analogous orientation test is the signed volume of a tetrahedron:

$$
\operatorname{orient3d}(a,b,c,d)
=
\det
\begin{bmatrix}
a_x-d_x & a_y-d_y & a_z-d_z\\
b_x-d_x & b_y-d_y & b_z-d_z\\
c_x-d_x & c_y-d_y & c_z-d_z
\end{bmatrix}
\tag{21.1.6}
$$

The signed Euclidean volume of the tetrahedron with vertices $a,b,c,d$ is:

$$
V(a,b,c,d)
=
\frac{1}{6}\operatorname{orient3d}(a,b,c,d)
\tag{21.1.7}
$$

Again, the sign is more important than the magnitude for many algorithms. It determines on which side of the oriented plane through $a,b,c$ the point $d$ lies, and it also determines whether a tetrahedral element has the prescribed orientation. In a finite element or finite volume code, an incorrect sign may reverse a cell, corrupt normal directions, invalidate element mappings, or destroy conservation across a shared face.

This distinction between representation and decision is central. Coordinates, connectivity, and metrics describe geometric data. Predicates decide how the data are interpreted. A mesh can store vertices and cells correctly, but if orientation, containment, or intersection tests are evaluated inconsistently, the resulting topology can be wrong. Such errors are discrete rather than smooth: a single incorrect sign may change a triangulation, misclassify a point, remove a boundary intersection, or generate a nonmanifold cell adjacency. This is why recent work on filtered and exact predicates, exact mesh arrangements, and exact mesh CSG treats robustness as a fundamental design issue rather than as a numerical afterthought (Bartels, Fisikopoulos and Weiser, 2023; Guo and Fu, 2024; Lévy, 2025).

### Rust Implementation

Following the discussion in Section 21.1.1 on geometric representation and orientation predicates, Program 21.1.1 presents a practical implementation of points, vectors, Euclidean metrics, and orientation tests in two and three dimensions. The program demonstrates how geometric information is transformed into computational decision procedures through determinant-based predicates. In computational geometry, geometric algorithms frequently branch on the sign of an orientation test rather than on the magnitude of a numerical quantity itself. Consequently, seemingly elementary operations such as vector subtraction, inner products, and determinant evaluation become central components of mesh generation, point classification, finite element orientation checks, Delaunay triangulation, and intersection algorithms. The implementation illustrates this transition from geometric representation to geometric decision by constructing two-dimensional and three-dimensional orientation predicates corresponding to equations (21.1.4)–(21.1.7), together with supporting Euclidean vector operations. The program also highlights the importance of sign consistency in finite-precision geometry, where incorrect orientation evaluation may corrupt topology, reverse element orientation, or invalidate geometric mappings.

At the core of the implementation are the `Point2`, `Vector2`, `Point3`, and `Vector3` structures, which represent geometric objects in two- and three-dimensional Euclidean space. These structures encode the coordinate representations introduced in equations (21.1.1) and (21.1.2). The `vector_to` methods construct displacement vectors between points, thereby implementing the vector difference operation (v=q-p). This distinction between points and vectors is important in geometric computing because points represent locations, while vectors represent directional or metric relationships between locations.

The `dot` and `norm` methods implement the Euclidean inner product and Euclidean norm defined in equation (21.1.3). The dot product computes the metric interaction between vectors, while the norm evaluates vector magnitude through the induced Euclidean metric. These operations are fundamental in computational geometry because they underlie distance computation, orthogonality testing, projection methods, angle estimation, and many mesh-quality measures used in finite element and finite volume discretizations.

The `cross` method defined for `Vector3` computes the three-dimensional vector cross product. This operation produces a vector orthogonal to the plane spanned by two input vectors and is used internally in the evaluation of the three-dimensional orientation determinant from equation (21.1.6). In geometric algorithms, cross products frequently appear in surface-normal construction, oriented area computation, polygon winding analysis, and finite-volume flux calculations.

The function `orient2d` implements the determinant-based orientation predicate introduced in equations (21.1.4) and (21.1.5). Given three planar points (a), (b), and (c), the function evaluates the signed area determinant associated with the oriented triangle. The sign of this determinant determines whether the ordered triple is positively oriented, negatively oriented, or collinear. This single predicate forms the basis of many geometric algorithms, including segment intersection testing, convex hull construction, polygon orientation checks, Delaunay edge flips, and planar point-location procedures. The accompanying helper function `classify_sign` converts the determinant value into a discrete orientation classification using a small tolerance to account for floating-point roundoff near degenerate configurations.

The function `orient3d` implements the three-dimensional orientation determinant from equation (21.1.6). The determinant is evaluated through the scalar triple product of vectors formed relative to the reference point (d). The resulting sign determines on which side of the oriented plane through (a), (b), and (c) the point (d) lies. The companion function `signed_tetrahedron_volume` then computes the signed Euclidean tetrahedral volume defined in equation (21.1.7) by dividing the orientation determinant by six. These operations are central in tetrahedral mesh generation, finite element orientation verification, geometric intersection algorithms, and volume-based discretizations. In practical scientific-computing codes, incorrect orientation signs can invert elements, reverse normal directions, and destroy conservation properties across shared interfaces.

The `main` function demonstrates the complete workflow of geometric representation and orientation decision in both two and three dimensions. It begins by constructing sample points and vectors, then evaluates Euclidean dot products and norms to verify the metric operations. The program next tests the two-dimensional orientation predicate on configurations corresponding to left-turn, right-turn, and collinear arrangements relative to an oriented segment. It then evaluates the three-dimensional orientation determinant and signed tetrahedral volume for points lying above, below, and on a reference plane. The printed classifications demonstrate how determinant signs are converted into discrete geometric decisions. The final metric examples verify the Euclidean structure implemented by the vector operations. Together, these demonstrations illustrate how low-level geometric predicates form the computational foundation for higher-level algorithms in mesh generation, finite elements, computational topology, and scientific simulation.

```rust
// Program 21.1.1: Point, Vector, and Orientation Predicates in Two and Three Dimensions
//
// Problem statement:
// Implement the basic geometric objects and decision predicates introduced in
// Section 21.1.1. The program represents points and vectors in two and three
// dimensions, computes vector differences, dot products, Euclidean norms,
// the two-dimensional orientation determinant, the three-dimensional
// orientation determinant, and the signed volume of a tetrahedron.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Vector2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy, Debug)]
struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy, Debug)]
enum Orientation {
    Positive,
    Negative,
    Degenerate,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn vector_to(self, other: Point2) -> Vector2 {
        Vector2 {
            x: other.x - self.x,
            y: other.y - self.y,
        }
    }
}

impl Vector2 {
    fn dot(self, other: Vector2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
}

impl Point3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn vector_to(self, other: Point3) -> Vector3 {
        Vector3 {
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl Vector3 {
    fn dot(self, other: Vector3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn cross(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

fn classify_sign(value: f64, tolerance: f64) -> Orientation {
    if value > tolerance {
        Orientation::Positive
    } else if value < -tolerance {
        Orientation::Negative
    } else {
        Orientation::Degenerate
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
}

fn orient3d(a: Point3, b: Point3, c: Point3, d: Point3) -> f64 {
    let ad = d.vector_to(a);
    let bd = d.vector_to(b);
    let cd = d.vector_to(c);

    ad.dot(bd.cross(cd))
}

fn signed_tetrahedron_volume(a: Point3, b: Point3, c: Point3, d: Point3) -> f64 {
    orient3d(a, b, c, d) / 6.0
}

fn main() {
    let tolerance = 1.0e-12;

    let p = Point2::new(1.0, 2.0);
    let q = Point2::new(4.0, 6.0);
    let v = p.vector_to(q);

    println!("Geometry as Representation and Decision");
    println!("=======================================");
    println!();

    println!("Two-Dimensional Point and Vector Data");
    println!("-------------------------------------");
    println!("p                  = ({:.6}, {:.6})", p.x, p.y);
    println!("q                  = ({:.6}, {:.6})", q.x, q.y);
    println!("q - p              = ({:.6}, {:.6})", v.x, v.y);
    println!("(q - p) dot (q - p)= {:.6}", v.dot(v));
    println!("||q - p||_2        = {:.6}", v.norm());
    println!();

    let a2 = Point2::new(0.0, 0.0);
    let b2 = Point2::new(2.0, 0.0);
    let c2_left = Point2::new(1.0, 1.0);
    let c2_right = Point2::new(1.0, -1.0);
    let c2_collinear = Point2::new(1.0, 0.0);

    let tests_2d = [
        ("left of segment", c2_left),
        ("right of segment", c2_right),
        ("collinear", c2_collinear),
    ];

    println!("Two-Dimensional Orientation Predicate");
    println!("-------------------------------------");
    println!("Segment from a = ({:.1}, {:.1}) to b = ({:.1}, {:.1})", a2.x, a2.y, b2.x, b2.y);
    println!("{:>18} {:>18} {:>18}", "case", "orient2d", "classification");

    for (label, c) in tests_2d {
        let det = orient2d(a2, b2, c);
        let class = classify_sign(det, tolerance);

        println!("{:>18} {:>18.10} {:>18?}", label, det, class);
    }

    println!();

    let a3 = Point3::new(0.0, 0.0, 0.0);
    let b3 = Point3::new(1.0, 0.0, 0.0);
    let c3 = Point3::new(0.0, 1.0, 0.0);
    let d3_above = Point3::new(0.0, 0.0, 1.0);
    let d3_below = Point3::new(0.0, 0.0, -1.0);
    let d3_coplanar = Point3::new(0.25, 0.25, 0.0);

    let tests_3d = [
        ("above plane", d3_above),
        ("below plane", d3_below),
        ("coplanar", d3_coplanar),
    ];

    println!("Three-Dimensional Orientation Predicate");
    println!("---------------------------------------");
    println!("Plane through a = (0,0,0), b = (1,0,0), c = (0,1,0)");
    println!(
        "{:>18} {:>18} {:>18} {:>18}",
        "case", "orient3d", "volume", "classification"
    );

    for (label, d) in tests_3d {
        let det = orient3d(a3, b3, c3, d);
        let volume = signed_tetrahedron_volume(a3, b3, c3, d);
        let class = classify_sign(det, tolerance);

        println!(
            "{:>18} {:>18.10} {:>18.10} {:>18?}",
            label, det, volume, class
        );
    }

    println!();

    let r = Point3::new(1.0, 2.0, 3.0);
    let s = Point3::new(4.0, 6.0, 8.0);
    let w = r.vector_to(s);

    println!("Three-Dimensional Metric Check");
    println!("------------------------------");
    println!("r                  = ({:.6}, {:.6}, {:.6})", r.x, r.y, r.z);
    println!("s                  = ({:.6}, {:.6}, {:.6})", s.x, s.y, s.z);
    println!("s - r              = ({:.6}, {:.6}, {:.6})", w.x, w.y, w.z);
    println!("(s - r) dot (s - r)= {:.6}", w.dot(w));
    println!("||s - r||_2        = {:.6}", w.norm());
}
```

Program 21.1.1 demonstrates how geometric representation and geometric decision procedures are integrated in computational geometry. The implementation shows that orientation predicates are not merely auxiliary calculations, but discrete logical operations whose signs determine topological structure, geometric consistency, and algorithmic correctness. This reflects the central theme of Section 21.1.1: computational geometry depends not only on storing coordinates and connectivity, but also on evaluating geometric predicates reliably and consistently.

The two-dimensional and three-dimensional orientation tests illustrate how determinant-based predicates convert continuous geometric configurations into discrete computational decisions. In two dimensions, the sign of the orientation determinant determines left-versus-right classification relative to an oriented segment. In three dimensions, the orientation determinant determines sidedness relative to an oriented plane and simultaneously defines the signed volume of a tetrahedron. These operations are fundamental in triangulations, convex hulls, finite element mappings, point-location algorithms, and conservative geometric discretizations.

The implementation also highlights the role of finite precision in geometric computing. Near degenerate configurations, determinant values may approach zero, making classification sensitive to roundoff error. Even a single incorrect sign may alter mesh topology, invalidate geometric adjacency relations, or produce inverted elements. The tolerance-based classification used in the program therefore serves as a simplified illustration of the robustness issues discussed later in Section 21.1.4, where filtered and exact predicates are introduced as more reliable approaches for large-scale geometric computation.

## 21.1.2. Simplices, Meshes, and Point Clouds

The most important geometric building block for many numerical methods is the simplex. In dimension $d$, a simplex is the convex hull of $d+1$ affinely independent vertices $v_0,\ldots,v_d \in \mathbb{R}^d$:

$$
T
=
\operatorname{conv}\{v_0,\ldots,v_d\}
=
\left\{
\sum_{i=0}^{d}\lambda_i v_i
:\lambda_i \ge 0,\;
\sum_{i=0}^{d}\lambda_i = 1
\right\}
\tag{21.1.8}
$$

The coefficients $\lambda_i$ are the barycentric coordinates of a point $x \in T$. Thus,

$$
x=\sum_{i=0}^{d}\lambda_i v_i,
\qquad
\sum_{i=0}^{d}\lambda_i=1
\tag{21.1.9}
$$

For a triangle in $\mathbb{R}^2$, barycentric coordinates are ratios of signed subareas; for a tetrahedron in $\mathbb{R}^3$, they are ratios of signed subvolumes. In computational practice, these coordinates perform several tasks at once. They test whether a point lies inside a simplex, define linear finite element shape functions, provide interpolation weights, and map physical coordinates to reference coordinates. The same mathematical object therefore connects geometry, approximation theory, and numerical linear algebra.

A mesh is a finite collection of geometric cells together with compatibility relations on their lower-dimensional entities. In a simplicial mesh, the cells are intervals in one dimension, triangles in two dimensions, and tetrahedra in three dimensions. More generally, a mesh may contain quadrilaterals, hexahedra, prisms, pyramids, polygons, or polyhedra. If $\mathcal{T}_h$ denotes a mesh of a domain $\Omega \subset \mathbb{R}^d$, then the cells $K \in \mathcal{T}_h$ usually satisfy:

$$
\overline{\Omega}
=
\bigcup_{K\in\mathcal{T}_h} \overline{K}
\tag{21.1.10}
$$

with intersections between distinct cells restricted to shared lower-dimensional entities such as vertices, edges, or faces. A conforming mesh is one in which these intersections are topologically consistent. A nonconforming or cut-cell mesh relaxes this condition, but then the geometric intersection logic becomes even more important.

In finite element and finite volume methods, mesh quality directly affects interpolation accuracy, matrix conditioning, conservation, and solver robustness. For a triangular element $K$, typical quality indicators include the minimum angle,

$$\theta_{\min}(K)=\min_{1\le i\le 3}\theta_i \tag{21.1.11}$$

and ratios involving the inradius $r_K$, circumradius $R_K$, and diameter $h_K$. For example,

$$q_K = \frac{2r_K}{R_K}\tag{21.1.12}$$

is close to one for a well-shaped equilateral triangle and deteriorates toward zero for highly distorted elements. Such measures are not merely geometric aesthetics. They control the stability of local coordinate maps and the conditioning of element matrices. Modern finite element mesh generation therefore treats geometry, topology, and numerical quality as coupled requirements (Lei et al., 2024; Zou et al., 2024; Ji et al., 2024).

A point cloud is different from a mesh. It is an unordered finite subset,

$$P={p_1,p_2,\ldots,p_N}\subset \mathbb{R}^d\tag{21.1.13}$$

with no explicit cell connectivity. Point clouds arise from measurement, sampling, particle methods, LiDAR acquisition, meshless approximation, and direct point-based analysis. Since connectivity is absent, geometric algorithms must infer locality by search. Typical queries include nearest-neighbor search,

$$
j^\ast=\arg\min_{1\le j\le N}\|x-p_j\|_2
\tag{21.1.14}
$$

and fixed-radius neighborhood search,

$$
\mathcal{N}_r(x)
=
\{p_j\in P:\|x-p_j\|_2\le r\}
\tag{21.1.15}
$$

The efficiency of these operations depends strongly on spatial indexes such as grids, voxel hashes, $k$-d trees, octrees, and bounding volume hierarchies. Recent point-cloud search literature emphasizes that memory locality, data layout, and dimension-dependent behavior are often as important as the formal asymptotic complexity of the search structure (Teuscher et al., 2025; Viñambres et al., 2026; Laso and Yermo, 2026).

### Rust Implementation

Following the discussion in Section 21.1.2 on simplices, meshes, barycentric coordinates, and point clouds, Program 21.1.2 presents a practical implementation of several geometric operations that form the foundation of finite element, interpolation, and meshless numerical methods. The program demonstrates how a simplex in two dimensions can be represented computationally through barycentric coordinates and how those coordinates simultaneously support point containment, interpolation, and geometric mapping. In addition, the implementation introduces elementary triangle-quality measures and basic point-cloud search operations, illustrating the distinction between connectivity-based mesh geometry and connectivity-free point-based geometry. The code therefore combines three central themes of Section 21.1.2: simplex representation through barycentric coordinates, mesh-quality evaluation through geometric metrics, and locality inference in point clouds through distance-based search. Together, these operations illustrate how computational geometry connects geometric representation, numerical approximation, and neighborhood structure in scientific computing.

At the core of the implementation are the `Point2`, `Triangle`, and `Barycentric` structures, which represent two-dimensional geometric primitives and simplex coordinate systems. The `Point2` structure stores Euclidean coordinates in $\mathbb{R}^2$, while the `Triangle` structure represents a two-dimensional simplex consisting of three affinely independent vertices. The `Barycentric` structure stores the coefficients $\lambda_0$, $\lambda_1$, and $\lambda_2$ introduced in equations (21.1.8) and (21.1.9). These coefficients define the affine coordinate representation of a point relative to the simplex and therefore connect geometry, interpolation, and finite element approximation through a unified coordinate system.

The function `orient2d` implements the determinant-based orientation predicate from equations (21.1.4) and (21.1.5). This predicate is used internally to compute signed subareas of triangles, which form the basis of barycentric-coordinate evaluation in two dimensions. The function `signed_area_twice` evaluates twice the signed area of the triangle, while the `area` function converts this quantity into the physical Euclidean area. These geometric area computations are essential because barycentric coordinates in triangles are naturally expressed as ratios of signed subareas.

The function `barycentric_coordinates` computes the simplex coordinates of a query point relative to a triangle. The implementation evaluates the signed subareas associated with the query point and normalizes them by the total triangle area. The resulting coordinates satisfy the affine partition-of-unity condition from equation (21.1.9). In computational practice, barycentric coordinates are exceptionally important because they serve several purposes simultaneously: they determine whether a point lies inside the simplex, define linear finite element shape functions, provide interpolation weights, and establish local coordinate mappings between physical and reference elements.

The function `contains_point` uses the barycentric coordinates to test simplex containment. A point is classified as lying inside the triangle when all barycentric coordinates are nonnegative and their sum is sufficiently close to one. This illustrates how geometric point-location algorithms are closely tied to affine coordinate systems in finite element and interpolation methods. The companion function `interpolate_linear` uses the barycentric coordinates as interpolation weights to evaluate a linear interpolant inside the simplex. This directly implements the interpolation framework described later in equation (21.1.19), where simplex coordinates define the local approximation basis.

The functions `edge_lengths`, `minimum_angle_degrees`, and `quality_ratio` implement elementary mesh-quality measures associated with equations (21.1.11) and (21.1.12). The edge lengths are first computed from Euclidean distances between vertices. The function `minimum_angle_degrees` then evaluates the smallest triangle angle using the law of cosines, while `quality_ratio` computes the ratio $2r_K/R_K$ involving the inradius and circumradius. These measures quantify geometric distortion and element quality. In finite element and finite volume methods, poor-quality elements can degrade interpolation accuracy, increase matrix ill-conditioning, and reduce numerical stability. The implementation therefore illustrates how geometric quality measures directly influence the reliability of scientific discretizations.

The functions `nearest_neighbor` and `fixed_radius_neighbors` implement the point-cloud search operations introduced in equations (21.1.14) and (21.1.15). The `nearest_neighbor` function performs a brute-force search for the closest point to a query location, while `fixed_radius_neighbors` identifies all points lying within a prescribed search radius. Although the implementation uses direct linear scans rather than advanced spatial indexing structures, it clearly demonstrates the computational role of locality inference in point-cloud geometry. In meshless methods, particle simulations, LiDAR analysis, and point-based approximation, such search procedures replace the explicit adjacency information that would normally be available in a mesh.

The `main` function demonstrates the complete workflow of simplex geometry, interpolation, mesh-quality evaluation, and point-cloud search. It begins by constructing a triangular simplex and evaluating the barycentric coordinates of a query point. The program then verifies that the coordinates satisfy the partition-of-unity condition and uses them to test whether the point lies inside the simplex. Next, a linear interpolant is evaluated from prescribed nodal values, demonstrating the use of barycentric coordinates as interpolation weights. The implementation then computes triangle-quality indicators, including edge lengths, minimum angle, and the ratio $2r_K/R_K$. Finally, the program constructs a simple point cloud and performs both nearest-neighbor and fixed-radius neighborhood searches. The printed output illustrates how the same geometric infrastructure supports interpolation, mesh quality analysis, and neighborhood inference in scientific computing applications.

```rust
// Program 21.1.2: Barycentric Coordinates, Triangle Quality, and Point-Cloud Search
//
// Problem statement:
// Implement the geometric operations introduced in Section 21.1.2. The program
// computes barycentric coordinates for a point in a triangle, uses them for
// point-in-simplex testing and linear interpolation, evaluates basic triangle
// quality measures, and demonstrates nearest-neighbor and fixed-radius search
// in an unordered point cloud.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    v0: Point2,
    v1: Point2,
    v2: Point2,
}

#[derive(Clone, Copy, Debug)]
struct Barycentric {
    lambda0: f64,
    lambda1: f64,
    lambda2: f64,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance_to(self, other: Point2) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl Triangle {
    fn new(v0: Point2, v1: Point2, v2: Point2) -> Self {
        Self { v0, v1, v2 }
    }

    fn signed_area_twice(self) -> f64 {
        orient2d(self.v0, self.v1, self.v2)
    }

    fn area(self) -> f64 {
        0.5 * self.signed_area_twice().abs()
    }

    fn barycentric_coordinates(self, p: Point2) -> Barycentric {
        let total = self.signed_area_twice();

        if total.abs() < 1.0e-14 {
            panic!("Degenerate triangle: barycentric coordinates are undefined.");
        }

        let lambda0 = orient2d(p, self.v1, self.v2) / total;
        let lambda1 = orient2d(self.v0, p, self.v2) / total;
        let lambda2 = orient2d(self.v0, self.v1, p) / total;

        Barycentric {
            lambda0,
            lambda1,
            lambda2,
        }
    }

    fn contains_point(self, p: Point2, tolerance: f64) -> bool {
        let b = self.barycentric_coordinates(p);

        b.lambda0 >= -tolerance
            && b.lambda1 >= -tolerance
            && b.lambda2 >= -tolerance
            && (b.lambda0 + b.lambda1 + b.lambda2 - 1.0).abs() <= tolerance
    }

    fn interpolate_linear(self, p: Point2, values: [f64; 3]) -> f64 {
        let b = self.barycentric_coordinates(p);

        b.lambda0 * values[0] + b.lambda1 * values[1] + b.lambda2 * values[2]
    }

    fn edge_lengths(self) -> [f64; 3] {
        [
            self.v1.distance_to(self.v2),
            self.v0.distance_to(self.v2),
            self.v0.distance_to(self.v1),
        ]
    }

    fn minimum_angle_degrees(self) -> f64 {
        let lengths = self.edge_lengths();

        let a = lengths[0];
        let b = lengths[1];
        let c = lengths[2];

        let angle0 = angle_from_edges(b, c, a);
        let angle1 = angle_from_edges(a, c, b);
        let angle2 = angle_from_edges(a, b, c);

        angle0.min(angle1).min(angle2).to_degrees()
    }

    fn quality_ratio(self) -> f64 {
        let lengths = self.edge_lengths();
        let a = lengths[0];
        let b = lengths[1];
        let c = lengths[2];

        let area = self.area();
        let semiperimeter = 0.5 * (a + b + c);

        if area <= 1.0e-14 || semiperimeter <= 1.0e-14 {
            return 0.0;
        }

        let inradius = area / semiperimeter;
        let circumradius = a * b * c / (4.0 * area);

        2.0 * inradius / circumradius
    }
}

impl Barycentric {
    fn sum(self) -> f64 {
        self.lambda0 + self.lambda1 + self.lambda2
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
}

fn angle_from_edges(side1: f64, side2: f64, opposite: f64) -> f64 {
    let numerator = side1 * side1 + side2 * side2 - opposite * opposite;
    let denominator = 2.0 * side1 * side2;

    let cosine = (numerator / denominator).clamp(-1.0, 1.0);
    cosine.acos()
}

fn nearest_neighbor(points: &[Point2], query: Point2) -> Option<(usize, f64)> {
    if points.is_empty() {
        return None;
    }

    let mut best_index = 0usize;
    let mut best_distance = points[0].distance_to(query);

    for (j, &point) in points.iter().enumerate().skip(1) {
        let distance = point.distance_to(query);

        if distance < best_distance {
            best_distance = distance;
            best_index = j;
        }
    }

    Some((best_index, best_distance))
}

fn fixed_radius_neighbors(points: &[Point2], query: Point2, radius: f64) -> Vec<(usize, f64)> {
    let mut neighbors = Vec::new();

    for (j, &point) in points.iter().enumerate() {
        let distance = point.distance_to(query);

        if distance <= radius {
            neighbors.push((j, distance));
        }
    }

    neighbors
}

fn main() {
    let tolerance = 1.0e-12;

    let triangle = Triangle::new(
        Point2::new(0.0, 0.0),
        Point2::new(2.0, 0.0),
        Point2::new(0.5, 1.5),
    );

    let query = Point2::new(0.75, 0.50);
    let nodal_values = [1.0, 3.0, 2.0];

    let bary = triangle.barycentric_coordinates(query);
    let inside = triangle.contains_point(query, tolerance);
    let interpolated = triangle.interpolate_linear(query, nodal_values);

    println!("Simplices, Barycentric Coordinates, and Point Clouds");
    println!("====================================================");
    println!();

    println!("Triangle Vertices");
    println!("-----------------");
    println!("v0 = ({:.6}, {:.6})", triangle.v0.x, triangle.v0.y);
    println!("v1 = ({:.6}, {:.6})", triangle.v1.x, triangle.v1.y);
    println!("v2 = ({:.6}, {:.6})", triangle.v2.x, triangle.v2.y);
    println!();

    println!("Barycentric Coordinates");
    println!("-----------------------");
    println!("query point x          = ({:.6}, {:.6})", query.x, query.y);
    println!("lambda_0               = {:.10}", bary.lambda0);
    println!("lambda_1               = {:.10}", bary.lambda1);
    println!("lambda_2               = {:.10}", bary.lambda2);
    println!("lambda_0+lambda_1+lambda_2 = {:.10}", bary.sum());
    println!("inside triangle        = {}", inside);
    println!("nodal values           = [{:.3}, {:.3}, {:.3}]", nodal_values[0], nodal_values[1], nodal_values[2]);
    println!("linear interpolant     = {:.10}", interpolated);
    println!();

    let lengths = triangle.edge_lengths();

    println!("Triangle Quality");
    println!("----------------");
    println!("area                    = {:.10}", triangle.area());
    println!("edge length opposite v0 = {:.10}", lengths[0]);
    println!("edge length opposite v1 = {:.10}", lengths[1]);
    println!("edge length opposite v2 = {:.10}", lengths[2]);
    println!("minimum angle           = {:.10} degrees", triangle.minimum_angle_degrees());
    println!("quality ratio 2r/R      = {:.10}", triangle.quality_ratio());
    println!();

    let point_cloud = vec![
        Point2::new(-0.20, 0.10),
        Point2::new(0.10, 0.20),
        Point2::new(0.70, 0.45),
        Point2::new(1.30, 0.60),
        Point2::new(1.80, 0.10),
        Point2::new(0.40, 1.20),
        Point2::new(1.10, 1.10),
    ];

    let cloud_query = Point2::new(0.80, 0.55);
    let radius = 0.55;

    println!("Point-Cloud Search");
    println!("------------------");
    println!("query point          = ({:.6}, {:.6})", cloud_query.x, cloud_query.y);
    println!("number of points     = {}", point_cloud.len());

    if let Some((index, distance)) = nearest_neighbor(&point_cloud, cloud_query) {
        let p = point_cloud[index];

        println!(
            "nearest neighbor     = index {}, point ({:.6}, {:.6}), distance {:.10}",
            index, p.x, p.y, distance
        );
    }

    let neighbors = fixed_radius_neighbors(&point_cloud, cloud_query, radius);

    println!("search radius        = {:.6}", radius);
    println!("radius neighbors     = {}", neighbors.len());

    for (index, distance) in neighbors {
        let p = point_cloud[index];

        println!(
            "  index {:>2}: point ({:>8.4}, {:>8.4}), distance = {:.10}",
            index, p.x, p.y, distance
        );
    }
}
```

Program 21.1.2 demonstrates how simplices, barycentric coordinates, mesh-quality measures, and point-cloud searches form interconnected components of computational geometry. The implementation shows that barycentric coordinates are not merely abstract affine coefficients, but practical computational objects that simultaneously support point containment, interpolation, and finite element coordinate mapping. This reflects the central theme of Section 21.1.2: the same geometric structures frequently serve multiple numerical purposes at once.

The triangle-quality computations illustrate how geometric distortion directly affects numerical reliability. Quantities such as minimum angle and the ratio $2r_K/R_K$ are not simply geometric diagnostics, but indicators of interpolation stability, conditioning behavior, and discretization robustness. In practical finite element and finite volume methods, poor-quality elements can degrade convergence and produce unstable numerical operators even when the underlying PDE discretization is formally correct.

The point-cloud search routines further illustrate the distinction between mesh-based and meshless geometric representations. Unlike meshes, point clouds contain no explicit adjacency information, so locality must be inferred through geometric search operations. Although the implementation uses straightforward brute-force algorithms, the same conceptual framework extends naturally to more advanced spatial data structures such as k-d trees, octrees, voxel hashes, and bounding volume hierarchies used in large-scale scientific computing. The program therefore provides a unified introduction to simplex geometry, mesh quality analysis, interpolation, and neighborhood search within a common computational framework.

## 21.1.3. Geometric Algorithms in Scientific Computing

The connection between computational geometry and scientific computing is clearest when one follows a typical simulation pipeline. A physical domain may originate as CAD geometry, measured point data, implicit surfaces, segmented images, or a combination of these sources. The domain must then be cleaned, intersected, meshed, indexed, and queried. Once the numerical method begins, the code must repeatedly answer geometric questions: Which cell contains this point? Which faces bound this control volume? Which neighboring particles lie within the smoothing radius? Does this segment intersect a boundary? What are the interpolation weights at this location? Is this element inverted? Are two cells adjacent along a valid common face?

These questions can be expressed mathematically as membership, intersection, projection, and optimization problems. For a closed set $D\subset \mathbb{R}^d$, point containment asks whether,

$$
x\in D,\qquad x\in \partial D,\qquad \text{or}\qquad x\notin D
\tag{21.1.16}
$$

For two geometric entities $A,B\subset \mathbb{R}^d$, intersection algorithms determine:

$$A\cap B \tag{21.1.17}$$

which may be empty, lower-dimensional, or full-dimensional. For remapping, cut-cell, and volume-of-fluid methods, the intersection volume,

$$|K\cap D|\tag{21.1.18}$$

may enter conservation laws directly. If this quantity is computed inconsistently, conservation can fail even when the differential equation discretization is otherwise sound. This is why robust polytope intersection and spherical intersection algorithms are essential in scientific-computing settings such as arbitrary-grid volume-of-fluid methods and conservative remapping on spherical grids (López and Hernández, 2024; Chen, Ullrich and Panetta, 2026).

Interpolation is another fundamental geometric operation. Given values $f(v_i)$ at the vertices of a simplex $T=\operatorname{conv}{v_0,\ldots,v_d}$, the linear interpolant at a point $x\in T$ is:

$$
I_T f(x)
=
\sum_{i=0}^{d}\lambda_i(x)f(v_i)
\tag{21.1.19}
$$

where $\lambda_i(x)$ are the barycentric coordinates from equation (1.1.9). This formula is simple, but its reliability depends on geometric point location, nondegenerate cells, correctly oriented elements, and stable coordinate evaluation. On polygonal or polyhedral cells, generalized barycentric coordinates extend the same idea beyond simplices, and recent work continues to develop stable coordinate constructions for finite element geometries and mean value coordinates (Dieci, Difonzo and Sukumar, 2024; Fuda and Hormann, 2024).

Triangulations and Voronoi diagrams provide another central bridge between geometry and numerical computation. Given a finite point set $P\subset\mathbb{R}^d$, the Voronoi cell of $p_i\in P$ is:

$$
V_i
=
\{x\in\mathbb{R}^{d}:\|x-p_i\|_2\le \|x-p_j\|_2\ \text{for all }j\}
\tag{21.1.20}
$$

The Delaunay triangulation is the geometric dual of the Voronoi diagram under suitable nondegeneracy assumptions. In two dimensions, the local Delaunay property is expressed by an empty-circumcircle condition, itself evaluated by a determinant predicate. Delaunay and Voronoi structures appear in scattered-data interpolation, control-volume construction, remeshing, nearest-neighbor graphs, and optimal-transport formulations. Their modern implementations include CPU, GPU, FPGA, parallel, and distributed variants, reflecting their continuing importance in large-scale scientific computing (Elshakhs et al., 2024; Gao and Chen, 2025; Lévy et al., 2025).

## 21.1.4. Robustness and Finite-Precision Geometry

The central numerical difficulty in computational geometry is that geometric algorithms often make discrete decisions using floating-point arithmetic. A small roundoff error in a distance or determinant may not merely perturb a final numerical value. It may change a branch decision. If a determinant whose exact value is positive is rounded to a negative value, an algorithm may reverse an orientation, choose the wrong side of a plane, flip the wrong edge, misclassify a point, or fail to detect an intersection.

Let $\widehat{\Delta}$ denote the floating-point evaluation of an exact determinant $\Delta$. A sign predicate is reliable only when:

$$
\operatorname{sign}(\widehat{\Delta})
=
\operatorname{sign}(\Delta)
\tag{21.1.21}
$$

The problematic regime is not where $|\Delta|$ is large, but where the configuration is nearly degenerate, so that:

$$|\Delta| \approx 0 \tag{21.1.22}$$

Near collinearity, coplanarity, cocircularity, or cosphericity, a purely floating-point predicate may be unable to certify the sign. A tolerance rule such as,

$$|\widehat{\Delta}| < \varepsilon\tag{21.1.23}$$

can be useful in engineering practice, but it is not a complete logical solution because different predicates may interact inconsistently. One local tolerance may classify a point as lying on an edge, while another related test classifies the same point as strictly outside a neighboring cell. Such inconsistencies can corrupt topology.

A modern robust approach is to separate fast approximate evaluation from certified decision. In a filtered predicate, the determinant is first evaluated in floating-point arithmetic together with an error bound. If the computed value is far enough from zero to certify its sign, the predicate returns immediately. If not, the computation is repeated using higher precision, expansion arithmetic, or exact arithmetic. Conceptually, the decision has the form:

$$
|\widehat{\Delta}|>E
\quad\Longrightarrow\quad
\operatorname{sign}(\Delta)=\operatorname{sign}(\widehat{\Delta})
\tag{21.1.24}
$$

where $E$ is a rigorous error bound for the floating-point evaluation. If this condition fails, exact fallback is invoked. This strategy preserves speed in ordinary nondegenerate cases while retaining correctness near degeneracy. Recent work on floating-point filters, exact mesh arrangements, and exact constructive solid geometry shows that such predicate-level robustness must be coordinated with construction and topology-level robustness across the whole geometric pipeline (Bartels, Fisikopoulos and Weiser, 2023; Guo and Fu, 2024; Lévy, 2025).

The guiding principle of this chapter is therefore that robust geometry is part of numerical reliability. A finite element matrix assembled on an inverted cell, a conservative remapping method using inconsistent intersection volumes, a point-cloud approximation using incorrect neighborhoods, or a Delaunay triangulation corrupted by a wrong incircle decision may fail for geometric reasons before the numerical method itself becomes the limiting factor. Computational geometry supplies the mathematical and algorithmic layer that prevents such failures. Subsequent sections develop this layer from primitive predicates and representations through polygonal algorithms, convexity, Delaunay and Voronoi structures, spatial search, mesh interpolation, point location, and finite-precision robustness.

# 21.2. Geometric Primitives and Predicates

The fundamental operations of computational geometry are built from a small number of primitive objects and sign tests. Points, vectors, line segments, triangles, tetrahedra, polygons, and cells provide the representation layer, while predicates provide the decision layer. A predicate is a function whose output is usually not a floating-point magnitude, but a discrete classification: positive, negative, zero; inside, outside, boundary; intersecting or disjoint; clockwise or counterclockwise. This makes predicates qualitatively different from ordinary numerical evaluations. A small floating-point error in a scalar value may only perturb an approximation, but a small error in the sign of a predicate may change the combinatorial structure of the algorithm.

For this reason, geometric predicates occupy a central place in scientific computing. They determine whether a finite element is properly oriented, whether a particle lies inside a smoothing neighborhood, whether a control volume face intersects an interface, whether a point belongs to a cell, whether a Delaunay edge should be flipped, and whether two mesh entities must be merged or kept distinct. Modern work on robust predicates and exact geometric processing therefore treats predicate evaluation as a primary numerical concern, especially when geometric decisions are made using finite-precision arithmetic (Bartels, Fisikopoulos and Weiser, 2023; Guo and Fu, 2024; Lévy, 2025).

## 21.2.1. Points, Vectors, Segments, and Affine Combinations

Let $p,q\in\mathbb{R}^d$ be points. The vector from $p$ to $q$ is:

$$
q-p=(q_1-p_1,\ldots,q_d-p_d)
\tag{21.2.1}
$$

The Euclidean distance between the two points is:

$$
\operatorname{dist}(p,q)=\|q-p\|_2
=
\left(\sum_{i=1}^{d}(q_i-p_i)^2\right)^{1/2}
\tag{21.2.2}
$$

A line segment with endpoints $a,b\in\mathbb{R}^d$ is the set,

$$
[a,b]
=
\{(1-t)a+tb:\ 0\le t\le 1\}
\tag{21.2.3}
$$

The parameter $t$ is an affine coordinate along the segment. If $x$ lies on the supporting line of the segment, then $x=a+t(b-a)$. In numerical algorithms, this representation is used for point-on-segment tests, segment intersection, clipping, and projection.

The orthogonal projection of a point $x\in\mathbb{R}^d$ onto the line through $a$ and $b$, assuming $a\ne b$, is obtained from:

$$
t^\ast
=
\frac{(x-a)\cdot(b-a)}{\|b-a\|_2^2}
\tag{21.2.4}
$$

The closest point on the segment $[a,b]$ is therefore:

$$
\Pi_{[a,b]}(x)
=
a+\operatorname{clamp}(t^\ast,0,1)(b-a)
\tag{21.2.5}
$$

where,

$$
\operatorname{clamp}(t,0,1)
=
\begin{cases}
0, & t<0,\\
t, & 0\le t\le 1,\\
1, & t>1
\end{cases}
\tag{21.2.6}
$$

This simple formula is widely used in distance queries, collision detection, closest-point projection, boundary recovery, and mesh-quality calculations. It also illustrates a recurring theme: even when a geometric computation produces a continuous value, its use in an algorithm often requires a discrete classification, here whether $t^\ast<0$, $0\le t^\ast\le 1$, or $t^\ast>1$.

More generally, an affine combination of points $v_0,\ldots,v_m\in\mathbb{R}^d$ has the form:

$$
x=\sum_{i=0}^{m}\lambda_i v_i,
\qquad
\sum_{i=0}^{m}\lambda_i=1
\tag{21.2.7}
$$

If additionally $\lambda_i\ge 0$ for all $i$, then $x$ is a convex combination. The convex hull of the points is:

$$
\operatorname{conv}\{v_0,\ldots,v_m\}
=
\left\{
\sum_{i=0}^{m}\lambda_i v_i:\lambda_i\ge 0,\;
\sum_{i=0}^{m}\lambda_i=1
\right\}
\tag{21.2.8}
$$

These formulas are the algebraic foundation for segments, triangles, tetrahedra, finite elements, interpolation weights, and point-containment tests.

Following the discussion in Section 21.2.1 on points, vectors, segments, affine coordinates, and convex combinations, Program 21.2.1 presents a practical implementation of several foundational geometric primitives and projection operators used throughout computational geometry and scientific computing. The program demonstrates how Euclidean vector operations support geometric decision procedures such as closest-point projection and segment classification. In particular, the implementation evaluates the affine segment parameter introduced in equation (21.2.4), clamps the parameter according to equation (21.2.6), and computes the closest-point projection defined in equation (21.2.5). These operations form the basis of many geometric algorithms used in collision detection, mesh processing, finite element boundary recovery, particle tracking, and geometric search. The program also introduces affine and convex combinations of points, illustrating how the same algebraic framework underlies interpolation, simplices, barycentric coordinates, and convex hull representations. Together, these examples demonstrate the close relationship between geometric representation and geometric classification in numerical algorithms.

At the core of the implementation are the `Point2`, `Vector2`, and `Segment2` structures, which represent geometric primitives in two-dimensional Euclidean space. The `Point2` structure stores Cartesian coordinates, while the `Vector2` structure represents displacement vectors between points. The method `vector_to` implements the vector difference operation from equation (21.2.1), producing the displacement vector (q-p). The `distance_to` method then computes the Euclidean distance introduced in equation (21.2.2) by evaluating the Euclidean norm of the displacement vector. These operations provide the metric foundation for all subsequent projection and interpolation calculations.

The methods `dot`, `norm_squared`, and `norm` implemented for the `Vector2` structure provide the Euclidean inner-product operations needed for projection and distance evaluation. The dot product defines the metric interaction between vectors, while the Euclidean norm evaluates vector magnitude. The auxiliary method `scale` performs scalar-vector multiplication and is used throughout the projection formulas to construct affine points along a segment. Together, these methods provide the linear algebra infrastructure required for affine geometry and projection-based computations.

The `Segment2` structure represents the line segment defined in equation (21.2.3). The method `direction` computes the segment direction vector (b-a), while `point_at` evaluates the affine segment parameterization\
\[\
x=a+t(b-a).\
\]\
This parameterization is fundamental in computational geometry because it provides a continuous coordinate representation of points lying on the supporting line of the segment. Many geometric algorithms, including clipping, collision detection, intersection testing, and closest-point evaluation, are built from this affine representation.

The function `project_point` implements the orthogonal projection formula introduced in equations (21.2.4)–(21.2.6). The method first computes the unconstrained affine projection parameter (t^\\ast) by projecting the vector (x-a) onto the segment direction (b-a). The parameter is then clamped to the interval (\[0,1\]) in order to restrict the projection to the finite segment rather than the infinite supporting line. The resulting clamped parameter determines the closest point on the segment, while the associated classification identifies whether the projection lies before endpoint (a), inside the segment interior, or beyond endpoint (b). This illustrates a recurring theme in computational geometry: a continuous numerical quantity is transformed into a discrete geometric classification used to guide algorithmic decisions.

The `ProjectionLocation` enumeration and `ProjectionResult` structure formalize the classification process associated with the projection parameter. Rather than returning only a numerical projection value, the implementation explicitly categorizes the geometric relationship between the query point and the segment. Such classifications are central in geometric computing because they determine contact states, neighborhood relations, boundary interactions, and mesh-topology updates.

The function `affine_combination` implements the affine coordinate representation introduced in equation (21.2.7). Given a collection of points and affine weights satisfying the partition-of-unity condition, the function evaluates the corresponding affine combination. This operation forms the algebraic basis for barycentric coordinates, interpolation, finite element shape functions, simplex mappings, and convex geometry. The companion function `is_convex_combination` then determines whether the affine weights additionally satisfy the nonnegativity conditions required for a convex combination and the convex hull definition in equation (21.2.8).

The `main` function demonstrates the complete workflow of geometric projection, affine representation, and convex classification. It begins by constructing sample points and vectors, then verifies the Euclidean distance computations associated with equations (21.2.1) and (21.2.2). The program next constructs a line segment and evaluates closest-point projections for three representative query points corresponding to projection before the first endpoint, projection inside the segment interior, and projection beyond the second endpoint. The reported affine parameters and classifications demonstrate how continuous projection values are converted into discrete geometric decisions. Finally, the implementation evaluates both convex and nonconvex affine combinations of points, illustrating the distinction between affine and convex coordinate systems. Together, these examples demonstrate how geometric primitives and affine representations form the computational foundation of many higher-level geometric algorithms.

```rust
// Program 21.2.1: Segment Projection, Closest-Point Query, and Affine Combinations
//
// Problem statement:
// Implement the basic geometric primitive operations introduced in Section
// 21.2.1. The program represents points and vectors in two dimensions,
// computes vector differences and Euclidean distances, projects a query point
// onto a line segment, clamps the affine segment parameter to [0, 1], classifies
// the projection location, and evaluates affine and convex combinations of
// points.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Vector2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Segment2 {
    a: Point2,
    b: Point2,
}

#[derive(Clone, Copy, Debug)]
enum ProjectionLocation {
    BeforeA,
    Interior,
    AfterB,
}

#[derive(Clone, Copy, Debug)]
struct ProjectionResult {
    t_unclamped: f64,
    t_clamped: f64,
    closest: Point2,
    distance: f64,
    location: ProjectionLocation,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn vector_to(self, other: Point2) -> Vector2 {
        Vector2 {
            x: other.x - self.x,
            y: other.y - self.y,
        }
    }

    fn add_vector(self, v: Vector2) -> Point2 {
        Point2 {
            x: self.x + v.x,
            y: self.y + v.y,
        }
    }

    fn distance_to(self, other: Point2) -> f64 {
        self.vector_to(other).norm()
    }
}

impl Vector2 {
    fn scale(self, alpha: f64) -> Vector2 {
        Vector2 {
            x: alpha * self.x,
            y: alpha * self.y,
        }
    }

    fn dot(self, other: Vector2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }
}

impl Segment2 {
    fn new(a: Point2, b: Point2) -> Self {
        Self { a, b }
    }

    fn direction(self) -> Vector2 {
        self.a.vector_to(self.b)
    }

    fn point_at(self, t: f64) -> Point2 {
        self.a.add_vector(self.direction().scale(t))
    }

    fn project_point(self, x: Point2) -> ProjectionResult {
        let ab = self.direction();
        let ax = self.a.vector_to(x);
        let denominator = ab.norm_squared();

        if denominator <= 1.0e-14 {
            panic!("Degenerate segment: endpoints are too close.");
        }

        let t_unclamped = ax.dot(ab) / denominator;
        let t_clamped = t_unclamped.clamp(0.0, 1.0);
        let closest = self.point_at(t_clamped);
        let distance = x.distance_to(closest);

        let location = if t_unclamped < 0.0 {
            ProjectionLocation::BeforeA
        } else if t_unclamped > 1.0 {
            ProjectionLocation::AfterB
        } else {
            ProjectionLocation::Interior
        };

        ProjectionResult {
            t_unclamped,
            t_clamped,
            closest,
            distance,
            location,
        }
    }
}

fn affine_combination(points: &[Point2], weights: &[f64]) -> Point2 {
    if points.len() != weights.len() {
        panic!("The number of points and weights must match.");
    }

    let weight_sum: f64 = weights.iter().sum();

    if (weight_sum - 1.0).abs() > 1.0e-12 {
        panic!("Affine weights must sum to one.");
    }

    let mut x = 0.0;
    let mut y = 0.0;

    for (&point, &weight) in points.iter().zip(weights.iter()) {
        x += weight * point.x;
        y += weight * point.y;
    }

    Point2 { x, y }
}

fn is_convex_combination(weights: &[f64], tolerance: f64) -> bool {
    let weight_sum: f64 = weights.iter().sum();

    weights.iter().all(|&w| w >= -tolerance) && (weight_sum - 1.0).abs() <= tolerance
}

fn main() {
    let p = Point2::new(1.0, 2.0);
    let q = Point2::new(5.0, 5.0);
    let pq = p.vector_to(q);

    println!("Geometric Primitives: Segments and Affine Combinations");
    println!("=======================================================");
    println!();

    println!("Point, Vector, and Distance Data");
    println!("--------------------------------");
    println!("p                    = ({:.6}, {:.6})", p.x, p.y);
    println!("q                    = ({:.6}, {:.6})", q.x, q.y);
    println!("q - p                = ({:.6}, {:.6})", pq.x, pq.y);
    println!("||q - p||_2           = {:.10}", pq.norm());
    println!("dist(p, q)           = {:.10}", p.distance_to(q));
    println!();

    let segment = Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 2.0));

    let query_points = [
        ("before endpoint a", Point2::new(-1.0, 1.0)),
        ("interior projection", Point2::new(2.0, 2.0)),
        ("after endpoint b", Point2::new(6.0, 1.0)),
    ];

    println!("Closest-Point Projection onto a Segment");
    println!("---------------------------------------");
    println!(
        "segment [a,b]        = [({:.6}, {:.6}), ({:.6}, {:.6})]",
        segment.a.x, segment.a.y, segment.b.x, segment.b.y
    );
    println!();
    println!(
        "{:>20} {:>14} {:>14} {:>22} {:>16} {:>16}",
        "case", "t*", "clamped t", "closest point", "distance", "location"
    );

    for (label, x) in query_points {
        let result = segment.project_point(x);

        println!(
            "{:>20} {:>14.8} {:>14.8} ({:>8.4}, {:>8.4}) {:>16.8} {:>16?}",
            label,
            result.t_unclamped,
            result.t_clamped,
            result.closest.x,
            result.closest.y,
            result.distance,
            result.location
        );
    }

    println!();

    let vertices = [
        Point2::new(0.0, 0.0),
        Point2::new(2.0, 0.0),
        Point2::new(0.0, 2.0),
    ];

    let convex_weights = [0.25, 0.50, 0.25];
    let affine_weights = [-0.20, 0.80, 0.40];

    let convex_point = affine_combination(&vertices, &convex_weights);
    let affine_point = affine_combination(&vertices, &affine_weights);

    println!("Affine and Convex Combinations");
    println!("------------------------------");
    println!(
        "convex weights       = [{:.2}, {:.2}, {:.2}]",
        convex_weights[0], convex_weights[1], convex_weights[2]
    );
    println!(
        "convex combination   = ({:.6}, {:.6})",
        convex_point.x, convex_point.y
    );
    println!(
        "is convex?           = {}",
        is_convex_combination(&convex_weights, 1.0e-12)
    );
    println!();
    println!(
        "affine weights       = [{:.2}, {:.2}, {:.2}]",
        affine_weights[0], affine_weights[1], affine_weights[2]
    );
    println!(
        "affine combination   = ({:.6}, {:.6})",
        affine_point.x, affine_point.y
    );
    println!(
        "is convex?           = {}",
        is_convex_combination(&affine_weights, 1.0e-12)
    );
}
```

Program 21.2.1 demonstrates how elementary geometric primitives support a wide range of computational geometry operations through affine representation and projection-based classification. The implementation illustrates that even simple operations such as vector subtraction, Euclidean distance evaluation, and segment projection become algorithmically significant when their outputs determine geometric decisions. This reflects the central theme of Section 21.2.1: geometric primitives are not merely passive data structures, but active computational objects that guide branching logic and topological interpretation in numerical algorithms.

The closest-point projection examples illustrate how affine coordinates naturally encode geometric relationships along a segment. The projection parameter (t^\\ast) simultaneously determines the location of the orthogonal projection and classifies whether the nearest point lies on the segment interior or at an endpoint. Such projection operators are fundamental in collision detection, closest-point search, geometric optimization, boundary recovery, particle tracking, and finite element contact algorithms.

The affine and convex combination computations further demonstrate how linear coordinate systems unify geometry and approximation theory. Affine combinations preserve the geometric structure of the supporting space, while convex combinations define the convex hull and ensure geometric containment. These same principles reappear throughout computational geometry in barycentric coordinates, simplex interpolation, finite element shape functions, and convex-hull algorithms. The program therefore provides a compact computational foundation for many of the higher-level geometric constructions developed later in the chapter.

## 21.2.2. Orientation Predicates in Two and Three Dimensions

The most important geometric predicates are determinant signs. In two dimensions, the orientation of three points $a,b,c\in\mathbb{R}^2$ is:

$$
\operatorname{orient2d}(a,b,c)
=
\det
\begin{bmatrix}
a_x-c_x & a_y-c_y\\
b_x-c_x & b_y-c_y
\end{bmatrix}
\tag{21.2.9}
$$

Equivalently,

$$
\operatorname{orient2d}(a,b,c)
=
(a_x-c_x)(b_y-c_y)
-
(a_y-c_y)(b_x-c_x)
\tag{21.2.10}
$$

The sign of this determinant gives the orientation of the ordered triple:

$$
\operatorname{sign}(\operatorname{orient2d}(a,b,c))
=
\begin{cases}
+1, & a,b,c \text{ are counterclockwise},\\
0, & a,b,c \text{ are collinear},\\
-1, & a,b,c \text{ are clockwise}
\end{cases}
\tag{21.2.11}
$$

Depending on the chosen ordering convention, the labels clockwise and counterclockwise may be reversed, but the essential invariant is the sign. The determinant is twice the signed area of the triangle:

$$
A_{\mathrm{signed}}(a,b,c)
=
\frac{1}{2}\operatorname{orient2d}(a,b,c)
\tag{21.2.12}
$$

Thus, the unsigned area is:

$$
A(a,b,c)
=
\frac{1}{2}\left|\operatorname{orient2d}(a,b,c)\right|
\tag{21.2.13}
$$

The same idea extends to polygon area. For a polygon with ordered vertices $p_0,p_1,\ldots,p_{n-1}$, where indices are understood modulo $n$, the signed area is:

$$
A_{\mathrm{poly}}
=
\frac{1}{2}\sum_{i=0}^{n-1}
\left(p_{i,x}p_{i+1,y}
-
p_{i+1,x}p_{i,y}\right)
\tag{21.2.14}
$$

This is the classical shoelace formula. Its sign encodes the orientation of the vertex ordering, while its magnitude gives the polygonal area for a simple polygon. In finite volume methods, polygonal clipping, remapping, and interface reconstruction, this formula and its determinant interpretation are repeatedly used to compute areas, centroids, and orientation-consistent boundary integrals.

In three dimensions, the orientation predicate is the signed volume determinant:

$$
\operatorname{orient3d}(a,b,c,d)
=
\det
\begin{bmatrix}
a_x-d_x & a_y-d_y & a_z-d_z\\
b_x-d_x & b_y-d_y & b_z-d_z\\
c_x-d_x & c_y-d_y & c_z-d_z
\end{bmatrix}
\tag{21.2.15}
$$

The signed volume of the tetrahedron with vertices $a,b,c,d$ is:

$$
V_{\mathrm{signed}}(a,b,c,d)
=
\frac{1}{6}\operatorname{orient3d}(a,b,c,d)
\tag{21.2.16}
$$

The sign determines on which side of the oriented plane through $a,b,c$ the point $d$ lies. If the determinant is zero, the four points are coplanar. In tetrahedral mesh generation and finite element assembly, the sign of (21.2.15) determines whether the local element orientation is valid. A negative element volume may indicate an inverted element, while a nearly zero volume indicates degeneracy or severe ill-conditioning.

The determinant form also shows why these predicates are sensitive near degeneracy. If three points are nearly collinear or four points are nearly coplanar, the exact determinant is close to zero. In floating-point arithmetic, the computed sign may then be unreliable unless special care is taken. Since orientation signs are branch conditions in algorithms, an incorrect sign may alter mesh topology, point classification, or intersection structure. This is the principal motivation for filtered and exact predicate evaluation (Bartels, Fisikopoulos and Weiser, 2023).

Following the discussion in Section 21.2.2 on determinant-based orientation predicates, signed areas, and signed volumes, Program 21.2.2 presents a practical implementation of the most important geometric sign tests used in computational geometry and scientific computing. The program demonstrates how determinant evaluations are converted into discrete geometric classifications in both two and three dimensions. In the planar case, the implementation evaluates the orientation determinant associated with equations (21.2.9)–(21.2.13), using its sign to distinguish counterclockwise, clockwise, and collinear point configurations while simultaneously computing signed and unsigned triangle areas. The program then extends the same determinant framework to polygon orientation through the shoelace formula of equation (21.2.14) and to tetrahedral orientation and signed volume through equations (21.2.15) and (21.2.16). These operations illustrate a central theme of computational geometry: determinant signs are not merely numerical quantities, but logical predicates that determine geometric topology, orientation consistency, and algorithmic branching decisions. The implementation also highlights the sensitivity of these predicates near degenerate configurations, where finite-precision arithmetic may affect the reliability of sign classification.

At the core of the implementation are the `Point2`, `Point3`, and `Vector3` structures, which represent geometric primitives in two- and three-dimensional Euclidean space. The `Point2` and `Point3` structures store Cartesian coordinates, while the `Vector3` structure supports the vector operations needed for three-dimensional determinant evaluation. The method `vector_to` constructs displacement vectors between points, thereby implementing the geometric difference operations used throughout the orientation predicates. These geometric primitives provide the representation layer upon which all subsequent determinant-based decision procedures are constructed.

The methods `dot` and `cross` implemented for the `Vector3` structure provide the linear algebra infrastructure needed for evaluating the three-dimensional orientation determinant. The cross product computes a vector orthogonal to the plane spanned by two input vectors, while the dot product evaluates the scalar triple product defining the signed tetrahedral volume. Together, these operations implement the determinant form appearing in equation (21.2.15). Such vector operations are fundamental throughout computational geometry, finite element assembly, surface-normal construction, and conservative finite-volume discretizations.

The function `orient2d` implements the two-dimensional orientation determinant introduced in equations (21.2.9) and (21.2.10). Given three planar points (a), (b), and (c), the function evaluates the signed determinant whose sign determines the orientation classification defined in equation (21.2.11). The helper function `classify_orient2d` converts the determinant into the discrete categories `Counterclockwise`, `Clockwise`, and `Collinear` using a finite tolerance to account for floating-point roundoff near degenerate configurations. This illustrates the distinction between continuous numerical evaluation and discrete geometric decision: the magnitude of the determinant is less important than the sign classification derived from it.

The functions `signed_triangle_area` and `unsigned_triangle_area` implement the geometric area relations from equations (21.2.12) and (21.2.13). The signed area preserves orientation information and therefore changes sign when the vertex ordering is reversed, while the unsigned area measures the geometric area independently of orientation. In computational geometry and finite-volume methods, signed areas are important because they encode orientation-consistent boundary traversal and conservative geometric integration.

The function `polygon_signed_area` implements the shoelace formula introduced in equation (21.2.14). The implementation traverses the polygon vertices cyclically and accumulates the determinant contributions associated with successive edges. The resulting signed area simultaneously determines the polygon orientation and computes its geometric area. The helper function `classify_polygon_winding` then converts the sign of the area into a polygon winding classification. Such orientation-consistent polygon area computations are widely used in clipping algorithms, interface reconstruction, remapping procedures, computational topology, and geometric conservation laws.

The function `orient3d` implements the three-dimensional orientation determinant from equation (21.2.15). The determinant is evaluated through a scalar triple product constructed relative to the reference point (d). The resulting sign determines whether the point (d) lies on one side of the oriented plane through (a), (b), and (c), on the opposite side, or exactly on the plane itself. The companion function `signed_tetrahedral_volume` then computes the signed tetrahedral volume defined in equation (21.2.16). The helper function `classify_orient3d` converts the determinant sign into the classifications `Positive`, `Negative`, and `Coplanar`. These operations are central in tetrahedral mesh generation, finite element orientation verification, geometric intersection algorithms, and conservative volume discretizations.

The `main` function demonstrates the complete workflow of determinant-based geometric predicates in both two and three dimensions. It begins by evaluating the planar orientation predicate for counterclockwise, clockwise, and collinear point configurations, simultaneously computing signed and unsigned triangle areas. The program next evaluates polygon signed areas for both original and reversed vertex orderings, illustrating how the sign of the shoelace formula encodes polygon winding orientation. The implementation then evaluates the three-dimensional orientation determinant and signed tetrahedral volume for points lying on opposite sides of an oriented plane and for a coplanar configuration. Finally, the program reports the finite-precision tolerance used in the sign classifications, emphasizing the numerical sensitivity of determinant predicates near degeneracy. Together, these examples illustrate how orientation determinants provide the decision-making foundation for many computational geometry algorithms.

```rust
// Program 21.2.2: Orientation, Area, Polygon Winding, and Tetrahedral Volume Predicates
//
// Problem statement:
// Implement the orientation predicates introduced in Section 21.2.2. The
// program evaluates two-dimensional orientation determinants, signed and
// unsigned triangle areas, polygon signed area by the shoelace formula,
// three-dimensional orientation determinants, and signed tetrahedral volumes.
// Predicate signs are converted into explicit enum classifications.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy, Debug)]
struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy, Debug)]
enum Orientation2D {
    Counterclockwise,
    Clockwise,
    Collinear,
}

#[derive(Clone, Copy, Debug)]
enum Orientation3D {
    Positive,
    Negative,
    Coplanar,
}

#[derive(Clone, Copy, Debug)]
enum PolygonWinding {
    Counterclockwise,
    Clockwise,
    Degenerate,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl Point3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn vector_to(self, other: Point3) -> Vector3 {
        Vector3 {
            x: other.x - self.x,
            y: other.y - self.y,
            z: other.z - self.z,
        }
    }
}

impl Vector3 {
    fn dot(self, other: Vector3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Vector3) -> Vector3 {
        Vector3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
}

fn classify_orient2d(value: f64, tolerance: f64) -> Orientation2D {
    if value > tolerance {
        Orientation2D::Counterclockwise
    } else if value < -tolerance {
        Orientation2D::Clockwise
    } else {
        Orientation2D::Collinear
    }
}

fn signed_triangle_area(a: Point2, b: Point2, c: Point2) -> f64 {
    0.5 * orient2d(a, b, c)
}

fn unsigned_triangle_area(a: Point2, b: Point2, c: Point2) -> f64 {
    signed_triangle_area(a, b, c).abs()
}

fn polygon_signed_area(vertices: &[Point2]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }

    let mut sum = 0.0;
    let n = vertices.len();

    for i in 0..n {
        let p = vertices[i];
        let q = vertices[(i + 1) % n];

        sum += p.x * q.y - q.x * p.y;
    }

    0.5 * sum
}

fn classify_polygon_winding(area: f64, tolerance: f64) -> PolygonWinding {
    if area > tolerance {
        PolygonWinding::Counterclockwise
    } else if area < -tolerance {
        PolygonWinding::Clockwise
    } else {
        PolygonWinding::Degenerate
    }
}

fn orient3d(a: Point3, b: Point3, c: Point3, d: Point3) -> f64 {
    let da = d.vector_to(a);
    let db = d.vector_to(b);
    let dc = d.vector_to(c);

    da.dot(db.cross(dc))
}

fn classify_orient3d(value: f64, tolerance: f64) -> Orientation3D {
    if value > tolerance {
        Orientation3D::Positive
    } else if value < -tolerance {
        Orientation3D::Negative
    } else {
        Orientation3D::Coplanar
    }
}

fn signed_tetrahedral_volume(a: Point3, b: Point3, c: Point3, d: Point3) -> f64 {
    orient3d(a, b, c, d) / 6.0
}

fn main() {
    let tolerance = 1.0e-12;

    println!("Orientation Predicates in Two and Three Dimensions");
    println!("==================================================");
    println!();

    let a = Point2::new(0.0, 0.0);
    let b = Point2::new(2.0, 0.0);

    let triangle_tests = [
        ("counterclockwise", Point2::new(1.0, 1.0)),
        ("clockwise", Point2::new(1.0, -1.0)),
        ("collinear", Point2::new(1.0, 0.0)),
    ];

    println!("Two-Dimensional Orientation and Triangle Area");
    println!("---------------------------------------------");
    println!("base points a = ({:.1}, {:.1}), b = ({:.1}, {:.1})", a.x, a.y, b.x, b.y);
    println!(
        "{:>18} {:>16} {:>18} {:>18} {:>20}",
        "case", "orient2d", "signed area", "unsigned area", "classification"
    );

    for (label, c) in triangle_tests {
        let det = orient2d(a, b, c);
        let signed_area = signed_triangle_area(a, b, c);
        let unsigned_area = unsigned_triangle_area(a, b, c);
        let class = classify_orient2d(det, tolerance);

        println!(
            "{:>18} {:>16.8} {:>18.8} {:>18.8} {:>20?}",
            label, det, signed_area, unsigned_area, class
        );
    }

    println!();

    let polygon_ccw = vec![
        Point2::new(0.0, 0.0),
        Point2::new(3.0, 0.0),
        Point2::new(3.0, 1.0),
        Point2::new(1.0, 2.0),
        Point2::new(0.0, 1.0),
    ];

    let polygon_cw: Vec<Point2> = polygon_ccw.iter().rev().copied().collect();

    let area_ccw = polygon_signed_area(&polygon_ccw);
    let area_cw = polygon_signed_area(&polygon_cw);

    println!("Polygon Signed Area and Winding");
    println!("-------------------------------");
    println!(
        "{:>18} {:>18} {:>20}",
        "polygon", "signed area", "winding"
    );
    println!(
        "{:>18} {:>18.8} {:>20?}",
        "original",
        area_ccw,
        classify_polygon_winding(area_ccw, tolerance)
    );
    println!(
        "{:>18} {:>18.8} {:>20?}",
        "reversed",
        area_cw,
        classify_polygon_winding(area_cw, tolerance)
    );

    println!();

    let p0 = Point3::new(0.0, 0.0, 0.0);
    let p1 = Point3::new(1.0, 0.0, 0.0);
    let p2 = Point3::new(0.0, 1.0, 0.0);

    let tetra_tests = [
        ("above plane", Point3::new(0.0, 0.0, 1.0)),
        ("below plane", Point3::new(0.0, 0.0, -1.0)),
        ("coplanar", Point3::new(0.25, 0.25, 0.0)),
    ];

    println!("Three-Dimensional Orientation and Tetrahedral Volume");
    println!("----------------------------------------------------");
    println!("oriented plane through p0 = (0,0,0), p1 = (1,0,0), p2 = (0,1,0)");
    println!(
        "{:>18} {:>18} {:>18} {:>18}",
        "case", "orient3d", "signed volume", "classification"
    );

    for (label, p3) in tetra_tests {
        let det = orient3d(p0, p1, p2, p3);
        let volume = signed_tetrahedral_volume(p0, p1, p2, p3);
        let class = classify_orient3d(det, tolerance);

        println!(
            "{:>18} {:>18.8} {:>18.8} {:>18?}",
            label, det, volume, class
        );
    }

    println!();

    println!("Finite-Precision Note");
    println!("---------------------");
    println!("The tolerance used for sign classification is {:.1e}.", tolerance);
    println!("Values with magnitude below this threshold are treated as degenerate.");
}
```

Program 21.2.2 demonstrates how determinant signs unify orientation testing, area computation, polygon winding analysis, and tetrahedral volume evaluation within a common computational framework. The implementation illustrates that the essential output of a geometric predicate is often not the numerical value itself, but the discrete sign classification derived from it. This reflects the central theme of Section 21.2.2: geometric predicates act as branch conditions that determine combinatorial structure, geometric topology, and algorithmic behavior.

The two-dimensional orientation and polygon-winding examples demonstrate how determinant signs encode orientation consistency in planar geometry. A reversal of vertex ordering changes the sign of the determinant while preserving the magnitude of the geometric area. Such orientation-sensitive computations are fundamental in convex hull algorithms, polygon clipping, Delaunay triangulation, interface reconstruction, and conservative finite-volume methods.

The three-dimensional orientation and signed-volume computations further illustrate the role of determinant predicates in tetrahedral geometry and finite element discretizations. The sign of the orientation determinant determines whether an element orientation is valid, inverted, or degenerate. Near coplanarity, however, the determinant magnitude becomes extremely small, making the sign sensitive to floating-point roundoff. The tolerance-based classifications implemented in the program therefore provide a simplified illustration of the robustness challenges discussed at the end of Section 21.2.2. In practical geometric computing, filtered predicates, adaptive precision arithmetic, and exact determinant evaluation are frequently required to ensure topological consistency and reliable geometric decision-making near degeneracy.

## 21.2.3. Incircle and Insphere Predicates

Orientation determines sidedness with respect to lines and planes. Delaunay and Voronoi algorithms require a second class of predicates: tests relative to circles and spheres. In two dimensions, the incircle predicate decides whether a point $d$ lies inside, on, or outside the circumcircle through three oriented points $a,b,c$. A common determinant form is:

$$
\operatorname{incircle}(a,b,c,d)
=
\det
\begin{bmatrix}
a_x-d_x & a_y-d_y & (a_x-d_x)^2+(a_y-d_y)^2\\
b_x-d_x & b_y-d_y & (b_x-d_x)^2+(b_y-d_y)^2\\
c_x-d_x & c_y-d_y & (c_x-d_x)^2+(c_y-d_y)^2
\end{bmatrix}
\tag{21.2.17}
$$

When $a,b,c$ are consistently oriented, the sign of this determinant determines whether $d$ lies inside or outside the circumcircle. The precise inside/outside sign depends on the orientation convention for $a,b,c$, so robust implementations usually combine the incircle test with an orientation convention. The degeneracy,

$$
\operatorname{incircle}(a,b,c,d)=0
\tag{21.2.18}
$$

means that the four points are cocircular.

The incircle predicate is central to Delaunay triangulation. In two dimensions, an edge is locally Delaunay if the vertex opposite the edge in one adjacent triangle does not lie inside the circumcircle of the other triangle. Edge flipping algorithms repeatedly apply this test to enforce the empty-circumcircle condition. Thus, Delaunay triangulation depends not only on combinatorial updates but also on reliable evaluation of determinant signs. This is one reason recent surveys and implementations of Delaunay triangulation place strong emphasis on predicate robustness and hardware-conscious implementation strategies (Elshakhs et al., 2024; Gao and Chen, 2025).

In three dimensions, the corresponding insphere predicate determines whether a point $e$ lies inside, on, or outside the sphere through four points $a,b,c,d$. One determinant form, translated by $e$, is:

$$
\operatorname{insphere}(a,b,c,d,e)
=
\det
\begin{bmatrix}
a_x-e_x & a_y-e_y & a_z-e_z & \|a-e\|_2^2\\
b_x-e_x & b_y-e_y & b_z-e_z & \|b-e\|_2^2\\
c_x-e_x & c_y-e_y & c_z-e_z & \|c-e\|_2^2\\
d_x-e_x & d_y-e_y & d_z-e_z & \|d-e\|_2^2
\end{bmatrix}
\tag{21.2.19}
$$

The degeneracy,

$$\operatorname{insphere}(a,b,c,d,e)=0\tag{21.2.20}$$

corresponds to five cospherical points. This case is particularly important in three-dimensional Delaunay triangulation because cospherical or nearly cospherical configurations can create ambiguous local connectivity. In practice, these ambiguities are common in structured, symmetric, CAD-derived, or adaptively refined data. Robust three-dimensional triangulation therefore requires a coordinated policy for orientation, insphere decisions, degeneracy handling, and connectivity updates.

### Rust Implementation

Following the discussion in Section 21.2.3 on incircle and insphere predicates, Program 21.2.3 presents a practical implementation of determinant-based geometric tests used in Delaunay triangulation and Voronoi geometry. While orientation predicates determine sidedness relative to lines and planes, the predicates implemented here determine sidedness relative to circles and spheres. In two dimensions, the program evaluates whether a query point lies inside, on, or outside the circumcircle defined by an oriented triangle using the determinant formulation of equations (21.2.17) and (21.2.18). In three dimensions, the implementation extends the same determinant framework to the insphere predicate of equations (21.2.19) and (21.2.20), determining whether a query point lies inside, on, or outside the circumsphere of an oriented tetrahedron. These predicates form the computational foundation of Delaunay triangulation algorithms, where local mesh connectivity decisions depend directly on the sign of determinant evaluations. The program also illustrates the strong coupling between orientation conventions and predicate interpretation, emphasizing why robust determinant evaluation is essential in geometric computing.

At the core of the implementation are the `Point2` and `Point3` structures, which represent Euclidean points in two and three dimensions. Each structure stores Cartesian coordinates and provides a `squared_distance_to` method that evaluates squared Euclidean distances. These squared distances appear directly in the determinant formulations of equations (21.2.17) and (21.2.19), where each row combines translated coordinate differences with squared radial distances relative to the query point. The use of translated coordinates improves numerical stability and reduces unnecessary arithmetic growth in determinant evaluation.

The enumeration types `CirclePosition`, `SpherePosition`, and `OrientationSign` provide discrete geometric classifications for predicate evaluation. Rather than returning only raw determinant values, the implementation converts determinant signs into explicit geometric interpretations such as `Inside`, `Outside`, `OnCircle`, and `OnSphere`. This reflects a central principle of computational geometry: determinant evaluations are important primarily because of the combinatorial decisions implied by their signs. The `OrientationSign` enumeration is used internally to standardize the interpretation of determinant values with respect to a finite tolerance near degeneracy.

The function `classify_sign` converts floating-point determinant values into sign classifications using a tolerance threshold. Values whose magnitude falls below the prescribed tolerance are treated as degenerate. This illustrates the numerical sensitivity discussed at the end of Section 21.2.3: when points become nearly cocircular or cospherical, determinant magnitudes approach zero, making the sign susceptible to floating-point roundoff. Even a single incorrect sign may alter local triangulation connectivity or invalidate Delaunay edge-flip decisions.

The functions `orient2d` and `orient3d` implement the determinant-based orientation predicates introduced earlier in equations (21.2.9)–(21.2.16). These orientation tests are essential because the interpretation of the incircle and insphere determinants depends on the orientation convention of the defining simplex. The sign of the incircle or insphere determinant alone is not sufficient; it must be interpreted consistently relative to the orientation of the triangle or tetrahedron. The implementation therefore combines orientation evaluation with determinant classification in order to produce orientation-independent inside/outside decisions.

The helper functions `determinant3` and `determinant4` evaluate (3\\times3) and (4\\times4) determinants explicitly. The three-dimensional determinant evaluator uses the standard cofactor expansion formula, while the four-dimensional determinant evaluator computes minors recursively through Laplace expansion. These functions provide the algebraic foundation for the incircle and insphere predicates. Although direct determinant expansion is not always the most computationally efficient strategy for large-scale geometric libraries, it clearly exposes the determinant structure of the predicates and directly reflects the mathematical formulas given in equations (21.2.17) and (21.2.19).

The function `incircle` implements the determinant formulation of the planar circumcircle predicate from equation (21.2.17). The coordinates of the defining triangle vertices are translated relative to the query point (d), and the resulting determinant determines whether the query point lies inside, on, or outside the circumcircle. The companion function `classify_incircle` combines this determinant with the orientation of the defining triangle in order to produce a consistent geometric classification independent of vertex ordering convention. If the determinant vanishes within the prescribed tolerance, the point is classified as cocircular, corresponding to equation (21.2.18).

The functions `insphere` and `classify_insphere` extend the same framework to the three-dimensional circumsphere predicate defined in equations (21.2.19) and (21.2.20). The determinant is evaluated using translated coordinates relative to the query point (e), together with squared Euclidean distances. The resulting sign determines whether the point lies inside or outside the sphere through the tetrahedron vertices. The classification again depends on the orientation convention of the tetrahedron, requiring orientation adjustment before the final geometric interpretation is produced. Degenerate values correspond to cospherical configurations, which are especially important in three-dimensional Delaunay triangulation because they may create ambiguous local connectivity.

The `main` function demonstrates the complete workflow of incircle and insphere classification. In the two-dimensional example, three vertices on the unit circle define a circumcircle, and the program classifies representative query points lying inside the circle, exactly on the circle, and outside the circle. The determinant values and squared distances are printed together with the resulting geometric classifications. The three-dimensional example repeats the same procedure for points relative to a unit sphere defined by tetrahedron vertices. Finally, the implementation reports the tolerance used for degeneracy handling, emphasizing the numerical sensitivity of determinant predicates near cocircular and cospherical configurations. Together, these examples illustrate how Delaunay-type geometric decisions are encoded through determinant signs and orientation-aware classification procedures.

```rust
// Program 21.2.3: Incircle and Insphere Predicates for Delaunay-Type Decisions
//
// Problem statement:
// Implement the incircle and insphere predicates introduced in Section 21.2.3.
// The program evaluates whether a query point lies inside, outside, or on the
// circumcircle of an oriented triangle, and whether a query point lies inside,
// outside, or on the circumsphere of an oriented tetrahedron. The predicates
// are evaluated using determinant formulas translated by the query point.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy, Debug)]
enum CirclePosition {
    Inside,
    Outside,
    OnCircle,
}

#[derive(Clone, Copy, Debug)]
enum SpherePosition {
    Inside,
    Outside,
    OnSphere,
}

#[derive(Clone, Copy, Debug)]
enum OrientationSign {
    Positive,
    Negative,
    Degenerate,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn squared_distance_to(self, other: Point2) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;

        dx * dx + dy * dy
    }
}

impl Point3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn squared_distance_to(self, other: Point3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;

        dx * dx + dy * dy + dz * dz
    }
}

fn classify_sign(value: f64, tolerance: f64) -> OrientationSign {
    if value > tolerance {
        OrientationSign::Positive
    } else if value < -tolerance {
        OrientationSign::Negative
    } else {
        OrientationSign::Degenerate
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
}

fn orient3d(a: Point3, b: Point3, c: Point3, d: Point3) -> f64 {
    let ax = a.x - d.x;
    let ay = a.y - d.y;
    let az = a.z - d.z;

    let bx = b.x - d.x;
    let by = b.y - d.y;
    let bz = b.z - d.z;

    let cx = c.x - d.x;
    let cy = c.y - d.y;
    let cz = c.z - d.z;

    ax * (by * cz - bz * cy)
        - ay * (bx * cz - bz * cx)
        + az * (bx * cy - by * cx)
}

fn determinant3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn determinant4(m: [[f64; 4]; 4]) -> f64 {
    let mut det = 0.0;

    for col in 0..4 {
        let mut minor = [[0.0; 3]; 3];

        for i in 1..4 {
            let mut minor_col = 0;

            for j in 0..4 {
                if j == col {
                    continue;
                }

                minor[i - 1][minor_col] = m[i][j];
                minor_col += 1;
            }
        }

        let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * m[0][col] * determinant3(minor);
    }

    det
}

fn incircle(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let ax = a.x - d.x;
    let ay = a.y - d.y;
    let bx = b.x - d.x;
    let by = b.y - d.y;
    let cx = c.x - d.x;
    let cy = c.y - d.y;

    determinant3([
        [ax, ay, ax * ax + ay * ay],
        [bx, by, bx * bx + by * by],
        [cx, cy, cx * cx + cy * cy],
    ])
}

fn classify_incircle(
    a: Point2,
    b: Point2,
    c: Point2,
    d: Point2,
    tolerance: f64,
) -> CirclePosition {
    let orientation = orient2d(a, b, c);
    let value = incircle(a, b, c, d);

    if orientation.abs() <= tolerance {
        panic!("Degenerate triangle: incircle classification is undefined.");
    }

    let orientation_adjusted_value = if orientation > 0.0 { value } else { -value };

    match classify_sign(orientation_adjusted_value, tolerance) {
        OrientationSign::Positive => CirclePosition::Inside,
        OrientationSign::Negative => CirclePosition::Outside,
        OrientationSign::Degenerate => CirclePosition::OnCircle,
    }
}

fn insphere(a: Point3, b: Point3, c: Point3, d: Point3, e: Point3) -> f64 {
    let ax = a.x - e.x;
    let ay = a.y - e.y;
    let az = a.z - e.z;
    let ar2 = ax * ax + ay * ay + az * az;

    let bx = b.x - e.x;
    let by = b.y - e.y;
    let bz = b.z - e.z;
    let br2 = bx * bx + by * by + bz * bz;

    let cx = c.x - e.x;
    let cy = c.y - e.y;
    let cz = c.z - e.z;
    let cr2 = cx * cx + cy * cy + cz * cz;

    let dx = d.x - e.x;
    let dy = d.y - e.y;
    let dz = d.z - e.z;
    let dr2 = dx * dx + dy * dy + dz * dz;

    determinant4([
        [ax, ay, az, ar2],
        [bx, by, bz, br2],
        [cx, cy, cz, cr2],
        [dx, dy, dz, dr2],
    ])
}

fn classify_insphere(
    a: Point3,
    b: Point3,
    c: Point3,
    d: Point3,
    e: Point3,
    tolerance: f64,
) -> SpherePosition {
    let orientation = orient3d(a, b, c, d);
    let value = insphere(a, b, c, d, e);

    if orientation.abs() <= tolerance {
        panic!("Degenerate tetrahedron: insphere classification is undefined.");
    }

    let orientation_adjusted_value = if orientation > 0.0 { value } else { -value };

    match classify_sign(orientation_adjusted_value, tolerance) {
        OrientationSign::Positive => SpherePosition::Inside,
        OrientationSign::Negative => SpherePosition::Outside,
        OrientationSign::Degenerate => SpherePosition::OnSphere,
    }
}

fn main() {
    let tolerance = 1.0e-12;

    println!("Incircle and Insphere Predicates");
    println!("================================");
    println!();

    let a = Point2::new(1.0, 0.0);
    let b = Point2::new(0.0, 1.0);
    let c = Point2::new(-1.0, 0.0);

    let circle_tests = [
        ("inside", Point2::new(0.0, 0.0)),
        ("on circle", Point2::new(0.0, -1.0)),
        ("outside", Point2::new(2.0, 0.0)),
    ];

    println!("Two-Dimensional Incircle Predicate");
    println!("----------------------------------");
    println!("Triangle vertices lie on the unit circle.");
    println!("orientation orient2d(a,b,c) = {:.10}", orient2d(a, b, c));
    println!();
    println!(
        "{:>12} {:>18} {:>18} {:>18}",
        "case", "incircle", "dist^2 to origin", "classification"
    );

    for (label, d) in circle_tests {
        let value = incircle(a, b, c, d);
        let position = classify_incircle(a, b, c, d, tolerance);

        println!(
            "{:>12} {:>18.10} {:>18.10} {:>18?}",
            label,
            value,
            d.squared_distance_to(Point2::new(0.0, 0.0)),
            position
        );
    }

    println!();

    let p0 = Point3::new(1.0, 0.0, 0.0);
    let p1 = Point3::new(0.0, 1.0, 0.0);
    let p2 = Point3::new(0.0, 0.0, 1.0);
    let p3 = Point3::new(-1.0, 0.0, 0.0);

    let sphere_tests = [
        ("inside", Point3::new(0.0, 0.0, 0.0)),
        ("on sphere", Point3::new(0.0, -1.0, 0.0)),
        ("outside", Point3::new(2.0, 0.0, 0.0)),
    ];

    println!("Three-Dimensional Insphere Predicate");
    println!("------------------------------------");
    println!("Tetrahedron vertices lie on the unit sphere.");
    println!(
        "orientation orient3d(p0,p1,p2,p3) = {:.10}",
        orient3d(p0, p1, p2, p3)
    );
    println!();
    println!(
        "{:>12} {:>18} {:>18} {:>18}",
        "case", "insphere", "dist^2 to origin", "classification"
    );

    for (label, e) in sphere_tests {
        let value = insphere(p0, p1, p2, p3, e);
        let position = classify_insphere(p0, p1, p2, p3, e, tolerance);

        println!(
            "{:>12} {:>18.10} {:>18.10} {:>18?}",
            label,
            value,
            e.squared_distance_to(Point3::new(0.0, 0.0, 0.0)),
            position
        );
    }

    println!();

    println!("Finite-Precision Note");
    println!("---------------------");
    println!("The tolerance used for cocircular and cospherical tests is {:.1e}.", tolerance);
    println!("Values below this threshold are classified as degenerate boundary cases.");
}
```

Program 21.2.3 demonstrates how incircle and insphere predicates extend orientation-based geometry from linear and planar sidedness tests to curved geometric structures such as circumcircles and circumspheres. The implementation illustrates that Delaunay triangulation algorithms depend fundamentally on reliable determinant evaluation because local connectivity updates are controlled directly by the signs of geometric predicates. This reflects the central theme of Section 21.2.3: geometric topology in triangulation algorithms is determined not only by combinatorial data structures, but also by robust numerical classification.

The incircle examples illustrate how the circumcircle predicate supports local Delaunay edge decisions. A point lying inside the circumcircle violates the empty-circumcircle condition and may trigger an edge flip during triangulation refinement. The sign of the determinant therefore directly controls mesh connectivity updates. Similarly, the insphere examples demonstrate how three-dimensional Delaunay algorithms depend on sphere-based geometric decisions to maintain valid tetrahedral connectivity.

The implementation also highlights the numerical challenges associated with cocircular and cospherical degeneracies. Near-degenerate configurations produce determinants whose magnitudes are extremely small, making the sign highly sensitive to finite-precision roundoff. The tolerance-based classifications used in the program therefore provide a simplified illustration of the robustness issues emphasized in modern geometric computing. In practical Delaunay triangulation software, adaptive precision arithmetic, filtered predicates, symbolic perturbation, and exact determinant evaluation are often required to maintain topological consistency and reliable connectivity decisions in the presence of nearly degenerate geometric configurations.

## 21.2.4. Barycentric Coordinates and Point-in-Simplex Tests

Barycentric coordinates provide a unified algebraic language for point containment, interpolation, finite element shape functions, and simplex geometry. Let:

$$
T=\operatorname{conv}\{v_0,v_1,\ldots,v_d\}\subset\mathbb{R}^{d}
\tag{21.2.21}
$$

be a $d$-simplex with affinely independent vertices. A point $x\in\mathbb{R}^d$ has barycentric coordinates $\lambda_0,\ldots,\lambda_d$ with respect to $T$ if:

$$
x=\sum_{i=0}^{d}\lambda_i v_i,
\qquad
\sum_{i=0}^{d}\lambda_i=1
\tag{21.2.22}
$$

The point lies inside the closed simplex if and only if:

$$\lambda_i\ge 0,\qquad i=0,\ldots,d \tag{21.2.23}$$

It lies in the relative interior if:

$$\lambda_i>0,\qquad i=0,\ldots,d \tag{21.2.24}$$

and it lies on the boundary if at least one barycentric coordinate is zero:

$$
\min_{0\le i\le d}\lambda_i=0
\quad\text{and}\quad
\lambda_j\ge 0\ \text{for all }j
\tag{21.2.25}
$$

For a triangle $T=(v_0,v_1,v_2)$, barycentric coordinates can be written as signed area ratios:

$$
\lambda_0(x)
=
\frac{\operatorname{orient2d}(x,v_1,v_2)}
{\operatorname{orient2d}(v_0,v_1,v_2)}
\tag{21.2.26}
$$

$$
\lambda_1(x)
=
\frac{\operatorname{orient2d}(v_0,x,v_2)}
{\operatorname{orient2d}(v_0,v_1,v_2)}
\tag{21.2.27}
$$

$$
\lambda_2(x)
=
\frac{\operatorname{orient2d}(v_0,v_1,x)}
{\operatorname{orient2d}(v_0,v_1,v_2)}
\tag{21.2.28}
$$

For a tetrahedron $T=(v_0,v_1,v_2,v_3)$, barycentric coordinates are signed volume ratios:

$$
\lambda_2(x)
=
\frac{\operatorname{orient2d}(v_0,v_1,x)}
{\operatorname{orient2d}(v_0,v_1,v_2)}
\tag{21.2.28}
$$

$$
\lambda_1(x)
=
\frac{\operatorname{orient3d}(v_0,x,v_2,v_3)}
{\operatorname{orient3d}(v_0,v_1,v_2,v_3)}
\tag{21.2.30}
$$

$$
\lambda_2(x)
=
\frac{\operatorname{orient3d}(v_0,v_1,x,v_3)}
{\operatorname{orient3d}(v_0,v_1,v_2,v_3)}
\tag{21.2.31}
$$

$$
\lambda_3(x)
=
\frac{\operatorname{orient3d}(v_0,v_1,v_2,x)}
{\operatorname{orient3d}(v_0,v_1,v_2,v_3)}
\tag{21.2.32}
$$

Again, the denominator must be nonzero. In finite element computations, these barycentric coordinates are precisely the linear shape functions on a simplex:

$$N_i(x)=\lambda_i(x),\qquad i=0,\ldots,d \tag{21.2.33}$$

Thus, for nodal data $u_i=u(v_i)$, the linear interpolant is:

$$I_T u(x) = \sum_{i=0}^d \lambda_i(x)u_i \tag{21.2.34}$$

The gradients of the barycentric coordinates are constant on a simplex, which is one reason linear simplicial elements are algebraically simple. If $B$ is the $d\times d$ matrix,

$$
B=
\begin{bmatrix}
v_1-v_0 & v_2-v_0 & \cdots & v_d-v_0
\end{bmatrix}
\tag{21.2.35}
$$

then any point $x$ can be represented as:

$$
x=v_0+B\mu,
\qquad
\mu=(\lambda_1,\ldots,\lambda_d)^T
\tag{21.2.36}
$$

Hence,

$$
\mu=B^{-1}(x-v_0),
\qquad
\lambda_0=1-\sum_{i=1}^{d}\lambda_i
\tag{21.2.37}
$$

This formulation connects barycentric coordinates to affine mappings from a reference simplex. It is also the form most directly used in finite element assembly, element-quality analysis, and point location.

Although simplex barycentric coordinates are classical, modern scientific computing increasingly uses generalized barycentric coordinates on polygonal and polyhedral cells. These coordinates extend the interpolation and containment role of $\lambda_i$ beyond simplices, but their stable evaluation is more delicate. Recent work on nonnegative moment coordinates and stable mean value coordinates confirms that barycentric-type coordinates remain an active topic when finite element geometries move beyond simple triangles and tetrahedra (Dieci, Difonzo and Sukumar, 2024; Fuda and Hormann, 2024).

## 21.2.5. Predicate Conditioning and Degeneracy

A geometric predicate is well conditioned when its sign is insensitive to small perturbations of the input data. It is ill conditioned when the exact value is close to zero. For the orientation predicate, the degenerate set in two dimensions is:

$$\operatorname{orient2d}(a,b,c)=0 \tag{21.2.38}$$

which means that $a,b,c$ are collinear. For the three-dimensional orientation predicate, the degenerate set is:

$$\operatorname{orient3d}(a,b,c,d)=0 \tag{21.2.39}$$

which means that the four points are coplanar. For the incircle and insphere predicates, the degenerate sets are cocircularity and cosphericity:

$$\operatorname{incircle}(a,b,c,d)=0 \tag{21.2.40}$$

$$\operatorname{insphere}(a,b,c,d,e)=0 \tag{21.2.41}$$

These degeneracies are not rare in scientific computing. Structured meshes, symmetric domains, CAD models, sampled surfaces, and generated point sets often contain exactly or nearly degenerate configurations. Even when the mathematical data are nondegenerate, floating-point representation may place the computation close to a degenerate manifold. If a predicate value $\Delta$ is evaluated in floating-point arithmetic as $\widehat{\Delta}$, then the predicate is safe only if the sign can be certified:

$$\operatorname{sign}(\widehat{\Delta}) = \operatorname{sign}(\Delta) \tag{21.2.42}$$

A simple tolerance test,

$$|\widehat{\Delta}| \le \varepsilon,\tag{21.2.43}$$

may detect some near-degenerate cases, but it does not by itself ensure consistency across the entire algorithm. The same point may be classified differently by adjacent cells, or two intersection tests may make incompatible decisions. This is why tolerance-based comparisons are best understood as local engineering devices rather than complete correctness guarantees.

A filtered predicate uses a more principled structure. Suppose an error bound $E$ satisfies:

$$|\widehat{\Delta}-\Delta|\le E \tag{21.2.44}$$

If,

$$|\widehat{\Delta}|>E \tag{21.2.45}$$

then $\widehat{\Delta}$ and $\Delta$ must have the same sign, so the predicate can return the floating-point sign safely:

$$\operatorname{sign}(\Delta)=\operatorname{sign}(\widehat{\Delta}) \tag{21.2.46}$$

If condition (21.2.45) fails, the predicate cannot certify the sign using the current arithmetic. It must then fall back to a more accurate computation, such as extended precision, expansion arithmetic, rational arithmetic, or exact symbolic evaluation. This staged strategy is the basis of fast robust predicates: most nondegenerate cases are decided quickly, while difficult near-degenerate cases are handled by exact or higher-precision fallback (Bartels, Fisikopoulos and Weiser, 2023).

The important point is that the cost of a predicate should not be judged only by its arithmetic formula. For a fixed dimension, orientation and incircle predicates have constant-size determinant expressions, so their formal arithmetic complexity is $O(1)$. However, certified predicate evaluation has a data-dependent cost. It is cheap when the determinant is far from zero and more expensive near degeneracy. This is the correct scientific-computing interpretation: robust predicates are designed so that exactness is paid for only when the geometry requires it.

## 21.2.6. Classification Types and Algorithmic Use

Many geometric predicates naturally return more than a boolean. A point-in-cell query, for example, should distinguish:

$$\text{Inside},\qquad \text{Outside},\qquad \text{Boundary} \tag{21.2.47}$$

A segment-intersection test should distinguish a proper crossing, endpoint contact, collinear overlap, and disjointness. A mesh-orientation check should distinguish positively oriented, negatively oriented, and degenerate cells. Encoding all of these cases as true or false loses information and often forces later code to guess the missing classification.

For a triangle (T), the barycentric test gives a precise classification:

$$
x\in \operatorname{int}(T)
\quad\Longleftrightarrow\quad
\lambda_0>0,\ \lambda_1>0,\ \lambda_2>0
\tag{21.2.48}
$$

$$
x\in \partial T
\quad\Longleftrightarrow\quad
\lambda_i\ge 0\ \text{for all }i
\ \text{and}\ 
\lambda_j=0\ \text{for at least one }j
\tag{21.2.49}
$$

$$
x\notin T
\quad\Longleftrightarrow\quad
\lambda_j<0\ \text{for at least one }j
\tag{21.2.50}
$$

The same logic applies to tetrahedra and higher-dimensional simplices. In implementation, however, exact equalities such as $\lambda_j=0$ are not reliable unless the coordinates have been computed through certified predicates or an explicitly documented tolerance policy. This again motivates separating mathematical classification from numerical evaluation.

For scientific computing, classification conventions must be deterministic. If a point lies on a shared face between two cells, the code must decide whether both cells report boundary membership, whether one cell owns the point by convention, or whether the point is passed to a higher-level tie-breaking rule. Without such conventions, point location, interpolation, conservative transfer, and mesh traversal may become nondeterministic. Exact and robust geometric processing papers emphasize the same principle at larger scale: local decisions must be coordinated so that the global structure remains topologically consistent (Guo and Fu, 2024; Lévy, 2025).

## 21.2.7. Implementation Perspective for Rust

A reliable Rust implementation should separate geometric primitives, predicates, and robustness policies. Points and vectors should be lightweight immutable data types. Predicates should be explicit functions or trait methods whose return values encode geometric classifications rather than raw booleans. Robustness should be a documented policy of the predicate layer, not a hidden collection of unrelated tolerances.

A useful conceptual structure is:

$$
\begin{aligned}
&\text{coordinates}
\longrightarrow
\text{primitive kernels}
\longrightarrow
\text{filtered predicates}
\longrightarrow
\text{topological algorithms}
\\
&\tag{21.2.51}
\end{aligned}
$$

The coordinate layer stores points, vectors, and scalar values. The primitive kernel layer evaluates dot products, determinants, distances, and affine coordinates. The predicate layer converts numerical values into certified signs or classifications. The topological layer performs higher-level operations such as segment intersection, polygon clipping, triangulation, mesh traversal, and point location.

This separation is particularly important in Rust because the type system can help prevent accidental mixing of concepts. A predicate can return an enum such as orientation sign or containment class rather than a floating-point number whose interpretation is left to the caller. Mesh algorithms can work with indices into contiguous storage rather than pointer-rich structures. Exact or filtered kernels can be introduced behind stable interfaces without rewriting every higher-level algorithm. These design choices are not merely software preferences. They reflect the mathematical structure of computational geometry, where small primitive decisions determine large combinatorial outcomes.

### Rust Implementation

Following the discussion in Section 21.2.4 on barycentric coordinates, simplex containment, and affine coordinate mappings, Program 21.2.4 presents a practical implementation of barycentric coordinate evaluation for triangles and tetrahedra together with point-in-simplex classification and finite element interpolation. The program demonstrates both determinant-based and affine-map formulations of barycentric coordinates, thereby connecting the geometric interpretation of signed area and signed volume ratios in equations (21.2.26)–(21.2.32) with the matrix-based affine mapping formulation of equations (21.2.35)–(21.2.37). Using these barycentric coordinates, the implementation classifies query points as inside, boundary, or outside points according to the conditions of equations (21.2.23)–(21.2.25) and equations (21.2.48)–(21.2.50). The same coordinates are then used as linear simplex shape functions through equations (21.2.33) and (21.2.34), illustrating the deep connection between computational geometry and finite element interpolation. The program also demonstrates how robust classification policies can be encoded explicitly through enums and tolerance-aware decision logic, reflecting the implementation perspective discussed later in Sections 21.2.5–21.2.7.

At the core of the implementation are the `Point2` and `Point3` structures, which represent geometric points in two- and three-dimensional Euclidean space. The `Triangle` and `Tetrahedron` structures store simplicial cells whose vertices define affine coordinate systems. These lightweight immutable data types form the primitive geometric layer from which barycentric coordinates, interpolation operators, and classification predicates are constructed. The separation between geometric primitives and classification logic follows the implementation philosophy described in equation (21.2.51), where coordinate storage, primitive kernels, predicates, and topological algorithms are treated as distinct layers.

The structures `Barycentric2` and `Barycentric3` encapsulate barycentric coordinates on triangles and tetrahedra. Each structure stores the simplex weights $\lambda_i$ and provides helper methods for computing coordinate sums, converting coordinates into arrays, and performing geometric classification. These structures provide a natural algebraic representation of the barycentric formulas introduced in equations (21.2.21)–(21.2.37). Since barycentric coordinates simultaneously encode interpolation weights and containment information, the same structures are reused throughout the implementation for point classification and finite element interpolation.

The enumeration `PointClassification` encodes the three geometric cases discussed in equations (21.2.47)–(21.2.50): `Inside`, `Boundary`, and `Outside`. Rather than returning a boolean predicate, the implementation preserves the full geometric classification. This reflects an important computational geometry principle emphasized in Section 21.2.6: reducing geometric decisions to simple true/false tests often discards important topological information and may force later code to infer missing cases incorrectly.

The methods `barycentric_by_area` and `barycentric_by_volume` implement the determinant-based barycentric coordinate formulas introduced in equations (21.2.26)–(21.2.32). In the two-dimensional case, the barycentric coordinates are evaluated as ratios of signed orientation determinants, corresponding geometrically to signed area ratios within the triangle. In the three-dimensional case, the coordinates are computed as signed volume ratios using the `orient3d` determinant. In both cases, the denominator corresponds to the signed measure of the simplex itself and must remain nonzero to avoid degeneracy. These implementations directly expose the geometric meaning of barycentric coordinates as normalized oriented subareas and subvolumes.

The methods `barycentric_by_affine_map` for both triangles and tetrahedra implement the affine-map formulation introduced in equations (21.2.35)–(21.2.37). In the triangular case, the method constructs the affine matrix $B$ associated with the simplex edges and solves a $2\times2$ linear system for the affine coordinates. In the tetrahedral case, the corresponding $3\times3$ affine map is solved using Cramer’s rule through the helper functions `determinant3` and `solve_3x3`. These implementations demonstrate the algebraic equivalence between determinant-based barycentric coordinates and affine coordinate systems derived from simplex mappings. This affine-map perspective is particularly important in finite element assembly, reference-element mappings, and mesh-quality analysis.

The methods `interpolate` implemented for both triangles and tetrahedra evaluate the linear finite element interpolant introduced in equation (21.2.34). The barycentric coordinates serve directly as the simplex shape functions defined in equation (21.2.33). Given nodal values $u_i$, the interpolated value is computed as a weighted barycentric combination of those nodal values. This demonstrates one of the most important computational roles of barycentric coordinates: they simultaneously provide containment logic, interpolation weights, and affine coordinate maps.

The helper function `classify_barycentric` implements the simplex containment conditions from equations (21.2.23)–(21.2.25) and equations (21.2.48)–(21.2.50). The function first verifies that the barycentric coordinates sum approximately to one, then checks whether any coordinate is negative beyond the prescribed tolerance. Points with all strictly positive coordinates are classified as interior points, points with one or more coordinates near zero are classified as boundary points, and points with at least one negative coordinate are classified as outside the simplex. The implementation therefore converts continuous barycentric coordinates into discrete geometric classifications while explicitly incorporating a tolerance-based numerical policy.

The functions `orient2d` and `orient3d` provide the determinant kernels used throughout the barycentric coordinate evaluations. These orientation predicates compute the signed areas and signed volumes required for determinant-based coordinate ratios. The helper functions `determinant3` and `solve_3x3` provide the linear algebra support required for affine coordinate evaluation in tetrahedral geometry. Together, these functions illustrate how barycentric coordinates depend fundamentally on determinant kernels and affine coordinate transformations.

The `main` function demonstrates the complete workflow of simplex-based geometric computation. The program first evaluates barycentric coordinates for points lying inside, on the boundary of, and outside a triangle, verifying the classification logic associated with equations (21.2.48)–(21.2.50). It then compares determinant-based and affine-map coordinate evaluations to confirm their equivalence and uses the barycentric coordinates as interpolation weights for a linear finite element interpolant. The same procedure is repeated for a tetrahedron using signed volume ratios and affine coordinate maps in three dimensions. Finally, the program reports the tolerance-based classification policy used throughout the implementation, illustrating how geometric predicates are converted into deterministic classification decisions in practical numerical software.

```rust
// Program 21.2.4: Barycentric Coordinates and Point-in-Simplex Classification
//
// Problem statement:
// Implement barycentric coordinates for triangles and tetrahedra, classify
// query points as Inside, Boundary, or Outside, and use the barycentric
// coordinates as linear finite element shape functions for interpolation.
// The program demonstrates signed area ratios in two dimensions, signed volume
// ratios in three dimensions, and affine-map based coordinate evaluation.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    v0: Point2,
    v1: Point2,
    v2: Point2,
}

#[derive(Clone, Copy, Debug)]
struct Tetrahedron {
    v0: Point3,
    v1: Point3,
    v2: Point3,
    v3: Point3,
}

#[derive(Clone, Copy, Debug)]
struct Barycentric2 {
    lambda0: f64,
    lambda1: f64,
    lambda2: f64,
}

#[derive(Clone, Copy, Debug)]
struct Barycentric3 {
    lambda0: f64,
    lambda1: f64,
    lambda2: f64,
    lambda3: f64,
}

#[derive(Clone, Copy, Debug)]
enum PointClassification {
    Inside,
    Boundary,
    Outside,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl Point3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

impl Triangle {
    fn new(v0: Point2, v1: Point2, v2: Point2) -> Self {
        Self { v0, v1, v2 }
    }

    fn barycentric_by_area(self, x: Point2) -> Barycentric2 {
        let denominator = orient2d(self.v0, self.v1, self.v2);

        if denominator.abs() <= 1.0e-14 {
            panic!("Degenerate triangle: barycentric coordinates are undefined.");
        }

        Barycentric2 {
            lambda0: orient2d(x, self.v1, self.v2) / denominator,
            lambda1: orient2d(self.v0, x, self.v2) / denominator,
            lambda2: orient2d(self.v0, self.v1, x) / denominator,
        }
    }

    fn barycentric_by_affine_map(self, x: Point2) -> Barycentric2 {
        let b00 = self.v1.x - self.v0.x;
        let b01 = self.v2.x - self.v0.x;
        let b10 = self.v1.y - self.v0.y;
        let b11 = self.v2.y - self.v0.y;

        let rhs0 = x.x - self.v0.x;
        let rhs1 = x.y - self.v0.y;

        let det = b00 * b11 - b01 * b10;

        if det.abs() <= 1.0e-14 {
            panic!("Degenerate triangle: affine map is singular.");
        }

        let lambda1 = (rhs0 * b11 - b01 * rhs1) / det;
        let lambda2 = (b00 * rhs1 - rhs0 * b10) / det;
        let lambda0 = 1.0 - lambda1 - lambda2;

        Barycentric2 {
            lambda0,
            lambda1,
            lambda2,
        }
    }

    fn interpolate(self, x: Point2, nodal_values: [f64; 3]) -> f64 {
        let b = self.barycentric_by_area(x);

        b.lambda0 * nodal_values[0]
            + b.lambda1 * nodal_values[1]
            + b.lambda2 * nodal_values[2]
    }
}

impl Tetrahedron {
    fn new(v0: Point3, v1: Point3, v2: Point3, v3: Point3) -> Self {
        Self { v0, v1, v2, v3 }
    }

    fn barycentric_by_volume(self, x: Point3) -> Barycentric3 {
        let denominator = orient3d(self.v0, self.v1, self.v2, self.v3);

        if denominator.abs() <= 1.0e-14 {
            panic!("Degenerate tetrahedron: barycentric coordinates are undefined.");
        }

        Barycentric3 {
            lambda0: orient3d(x, self.v1, self.v2, self.v3) / denominator,
            lambda1: orient3d(self.v0, x, self.v2, self.v3) / denominator,
            lambda2: orient3d(self.v0, self.v1, x, self.v3) / denominator,
            lambda3: orient3d(self.v0, self.v1, self.v2, x) / denominator,
        }
    }

    fn barycentric_by_affine_map(self, x: Point3) -> Barycentric3 {
        let b = [
            [
                self.v1.x - self.v0.x,
                self.v2.x - self.v0.x,
                self.v3.x - self.v0.x,
            ],
            [
                self.v1.y - self.v0.y,
                self.v2.y - self.v0.y,
                self.v3.y - self.v0.y,
            ],
            [
                self.v1.z - self.v0.z,
                self.v2.z - self.v0.z,
                self.v3.z - self.v0.z,
            ],
        ];

        let rhs = [
            x.x - self.v0.x,
            x.y - self.v0.y,
            x.z - self.v0.z,
        ];

        let mu = solve_3x3(b, rhs);

        let lambda1 = mu[0];
        let lambda2 = mu[1];
        let lambda3 = mu[2];
        let lambda0 = 1.0 - lambda1 - lambda2 - lambda3;

        Barycentric3 {
            lambda0,
            lambda1,
            lambda2,
            lambda3,
        }
    }

    fn interpolate(self, x: Point3, nodal_values: [f64; 4]) -> f64 {
        let b = self.barycentric_by_volume(x);

        b.lambda0 * nodal_values[0]
            + b.lambda1 * nodal_values[1]
            + b.lambda2 * nodal_values[2]
            + b.lambda3 * nodal_values[3]
    }
}

impl Barycentric2 {
    fn sum(self) -> f64 {
        self.lambda0 + self.lambda1 + self.lambda2
    }

    fn as_array(self) -> [f64; 3] {
        [self.lambda0, self.lambda1, self.lambda2]
    }

    fn classify(self, tolerance: f64) -> PointClassification {
        classify_barycentric(&self.as_array(), tolerance)
    }
}

impl Barycentric3 {
    fn sum(self) -> f64 {
        self.lambda0 + self.lambda1 + self.lambda2 + self.lambda3
    }

    fn as_array(self) -> [f64; 4] {
        [self.lambda0, self.lambda1, self.lambda2, self.lambda3]
    }

    fn classify(self, tolerance: f64) -> PointClassification {
        classify_barycentric(&self.as_array(), tolerance)
    }
}

fn classify_barycentric(lambda: &[f64], tolerance: f64) -> PointClassification {
    let sum: f64 = lambda.iter().sum();

    if (sum - 1.0).abs() > 10.0 * tolerance {
        return PointClassification::Outside;
    }

    if lambda.iter().any(|&value| value < -tolerance) {
        return PointClassification::Outside;
    }

    if lambda.iter().any(|&value| value.abs() <= tolerance) {
        PointClassification::Boundary
    } else {
        PointClassification::Inside
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
}

fn orient3d(a: Point3, b: Point3, c: Point3, d: Point3) -> f64 {
    let ax = a.x - d.x;
    let ay = a.y - d.y;
    let az = a.z - d.z;

    let bx = b.x - d.x;
    let by = b.y - d.y;
    let bz = b.z - d.z;

    let cx = c.x - d.x;
    let cy = c.y - d.y;
    let cz = c.z - d.z;

    ax * (by * cz - bz * cy)
        - ay * (bx * cz - bz * cx)
        + az * (bx * cy - by * cx)
}

fn determinant3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det_a = determinant3(a);

    if det_a.abs() <= 1.0e-14 {
        panic!("Singular 3 by 3 affine map.");
    }

    let a0 = [
        [b[0], a[0][1], a[0][2]],
        [b[1], a[1][1], a[1][2]],
        [b[2], a[2][1], a[2][2]],
    ];

    let a1 = [
        [a[0][0], b[0], a[0][2]],
        [a[1][0], b[1], a[1][2]],
        [a[2][0], b[2], a[2][2]],
    ];

    let a2 = [
        [a[0][0], a[0][1], b[0]],
        [a[1][0], a[1][1], b[1]],
        [a[2][0], a[2][1], b[2]],
    ];

    [
        determinant3(a0) / det_a,
        determinant3(a1) / det_a,
        determinant3(a2) / det_a,
    ]
}

fn print_barycentric2(label: &str, bary: Barycentric2, class: PointClassification) {
    println!(
        "{:>12} {:>14.8} {:>14.8} {:>14.8} {:>14.8} {:>12?}",
        label,
        bary.lambda0,
        bary.lambda1,
        bary.lambda2,
        bary.sum(),
        class
    );
}

fn print_barycentric3(label: &str, bary: Barycentric3, class: PointClassification) {
    println!(
        "{:>12} {:>12.8} {:>12.8} {:>12.8} {:>12.8} {:>12.8} {:>12?}",
        label,
        bary.lambda0,
        bary.lambda1,
        bary.lambda2,
        bary.lambda3,
        bary.sum(),
        class
    );
}

fn main() {
    let tolerance = 1.0e-12;

    println!("Barycentric Coordinates and Point-in-Simplex Tests");
    println!("==================================================");
    println!();

    let triangle = Triangle::new(
        Point2::new(0.0, 0.0),
        Point2::new(2.0, 0.0),
        Point2::new(0.0, 2.0),
    );

    let triangle_values = [1.0, 3.0, 5.0];

    let triangle_tests = [
        ("inside", Point2::new(0.50, 0.50)),
        ("boundary", Point2::new(1.00, 1.00)),
        ("outside", Point2::new(1.50, 1.00)),
    ];

    println!("Triangle Barycentric Coordinates by Signed Area Ratios");
    println!("------------------------------------------------------");
    println!(
        "{:>12} {:>14} {:>14} {:>14} {:>14} {:>12}",
        "case", "lambda0", "lambda1", "lambda2", "sum", "class"
    );

    for (label, x) in triangle_tests {
        let bary = triangle.barycentric_by_area(x);
        let class = bary.classify(tolerance);
        print_barycentric2(label, bary, class);
    }

    println!();

    let x = Point2::new(0.50, 0.50);
    let bary_area = triangle.barycentric_by_area(x);
    let bary_affine = triangle.barycentric_by_affine_map(x);
    let interpolated = triangle.interpolate(x, triangle_values);

    println!("Triangle Affine Map Check");
    println!("-------------------------");
    println!("query point                  = ({:.6}, {:.6})", x.x, x.y);
    println!(
        "area-ratio lambdas           = [{:.10}, {:.10}, {:.10}]",
        bary_area.lambda0, bary_area.lambda1, bary_area.lambda2
    );
    println!(
        "affine-map lambdas           = [{:.10}, {:.10}, {:.10}]",
        bary_affine.lambda0, bary_affine.lambda1, bary_affine.lambda2
    );
    println!(
        "nodal values                 = [{:.4}, {:.4}, {:.4}]",
        triangle_values[0], triangle_values[1], triangle_values[2]
    );
    println!("linear interpolant           = {:.10}", interpolated);
    println!();

    let tetrahedron = Tetrahedron::new(
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
    );

    let tetra_values = [2.0, 4.0, 6.0, 8.0];

    let tetra_tests = [
        ("inside", Point3::new(0.20, 0.20, 0.20)),
        ("boundary", Point3::new(0.50, 0.50, 0.00)),
        ("outside", Point3::new(0.60, 0.60, 0.20)),
    ];

    println!("Tetrahedron Barycentric Coordinates by Signed Volume Ratios");
    println!("-----------------------------------------------------------");
    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "case", "lambda0", "lambda1", "lambda2", "lambda3", "sum", "class"
    );

    for (label, x) in tetra_tests {
        let bary = tetrahedron.barycentric_by_volume(x);
        let class = bary.classify(tolerance);
        print_barycentric3(label, bary, class);
    }

    println!();

    let y = Point3::new(0.20, 0.20, 0.20);
    let bary_volume = tetrahedron.barycentric_by_volume(y);
    let bary_affine_3d = tetrahedron.barycentric_by_affine_map(y);
    let interpolated_3d = tetrahedron.interpolate(y, tetra_values);

    println!("Tetrahedron Affine Map Check");
    println!("----------------------------");
    println!(
        "query point                  = ({:.6}, {:.6}, {:.6})",
        y.x, y.y, y.z
    );
    println!(
        "volume-ratio lambdas         = [{:.10}, {:.10}, {:.10}, {:.10}]",
        bary_volume.lambda0,
        bary_volume.lambda1,
        bary_volume.lambda2,
        bary_volume.lambda3
    );
    println!(
        "affine-map lambdas           = [{:.10}, {:.10}, {:.10}, {:.10}]",
        bary_affine_3d.lambda0,
        bary_affine_3d.lambda1,
        bary_affine_3d.lambda2,
        bary_affine_3d.lambda3
    );
    println!(
        "nodal values                 = [{:.4}, {:.4}, {:.4}, {:.4}]",
        tetra_values[0], tetra_values[1], tetra_values[2], tetra_values[3]
    );
    println!("linear interpolant           = {:.10}", interpolated_3d);
    println!();

    println!("Classification Policy");
    println!("---------------------");
    println!("Inside   : all barycentric coordinates are strictly positive.");
    println!("Boundary : all coordinates are nonnegative and at least one is near zero.");
    println!("Outside  : at least one coordinate is negative beyond tolerance.");
    println!("tolerance = {:.1e}", tolerance);
}
```

Program 21.2.4 demonstrates how barycentric coordinates unify geometric containment, affine coordinate mapping, interpolation, and finite element shape functions within a single algebraic framework. The implementation shows that the same coordinates used to determine whether a point lies inside a simplex can also serve directly as interpolation weights and local coordinate maps. This dual geometric and numerical role is one reason barycentric coordinates occupy such a central position in finite element methods, computational geometry, mesh traversal, and scientific computing.

The determinant-based implementations illustrate the geometric interpretation of barycentric coordinates as signed area and signed volume ratios, while the affine-map implementations reveal their relationship to simplex coordinate transformations and local linear algebra. The exact agreement between these formulations confirms the equivalence between geometric orientation-based coordinates and affine coordinate systems derived from matrix inversion.

The classification examples further demonstrate how barycentric coordinates naturally support deterministic geometric decision-making. Interior points, boundary points, and exterior points are distinguished directly from the signs of the barycentric coordinates. However, the implementation also highlights the numerical sensitivity discussed in Section 21.2.5: exact equalities such as $\lambda_i=0$ are not generally reliable in floating-point arithmetic without a documented robustness policy. The tolerance-aware classification logic therefore provides a simplified illustration of the broader predicate-conditioning issues encountered in practical geometric software.

The modular organization of the code also reflects the implementation philosophy discussed in Section 21.2.7. Primitive kernels such as determinants and affine solves are separated from classification policies and topological interpretation. This design allows more advanced filtered predicates, exact arithmetic kernels, or generalized barycentric coordinates to be introduced later without changing the higher-level geometric algorithms built on top of the simplex coordinate framework.

# 21.3. Segment, Polygon, and Intersection Algorithms

Segment, polygon, and intersection algorithms form the first major layer above primitive predicates. They convert determinant signs, affine parameters, and boundary classifications into usable geometric operations: deciding whether two segments intersect, whether a point lies inside a polygon, whether a line cuts a convex cell, or whether a control volume must be clipped by an interface. In scientific computing, these operations are not merely planar drawing tools. They occur in finite-volume flux integration, conservative remapping, cut-cell methods, volume-of-fluid reconstruction, mesh intersection, CAD-to-analysis preprocessing, and geometric validation of unstructured meshes. The central idea is that most of these algorithms are compositions of the orientation, containment, and affine-coordinate predicates introduced in Section 21.2. The difficulty is that these compositions must remain consistent in the presence of boundary cases and finite-precision arithmetic. Recent work on polytope intersection, robust mesh arrangements, exact mesh CSG, and projective polygon algorithms reinforces the importance of deterministic and topology-preserving geometric classification in modern scientific-computing pipelines (López and Hernández, 2024; Guo and Fu, 2024; Lévy, 2025; Skala, 2025a; Skala, 2025b).

## 21.3.1. Segment Intersection and Orientation Logic

The intersection of two line segments is the canonical example of a compound geometric predicate. Let,

$$S_1=[a,b],\qquad S_2=[c,d],\qquad a,b,c,d\in\mathbb{R}^2  \tag{21.3.1}$$

The supporting lines are parametrized by:

$$x(s)=a+s(b-a),\qquad y(t)=c+t(d-c),\qquad 0\le s,t\le 1  \tag{21.3.2}$$

An intersection point satisfies:

$$a+s(b-a)=c+t(d-c) \tag{21.3.3}$$

Writing this as a $2\times 2$ linear system gives:

$$
\begin{bmatrix}
b_x-a_x & -(d_x-c_x)\\
b_y-a_y & -(d_y-c_y)
\end{bmatrix}
\begin{bmatrix}
s\\
t
\end{bmatrix}
=
\begin{bmatrix}
c_x-a_x\\
c_y-a_y
\end{bmatrix}
\tag{21.3.4}
$$

If the determinant,

$$
D
=
\det
\begin{bmatrix}
b_x-a_x & -(d_x-c_x)\\
b_y-a_y & -(d_y-c_y)
\end{bmatrix}
\tag{21.3.5}
$$

is nonzero, the two supporting lines are not parallel, and the unique line-line intersection can be tested by checking whether,

$$0\le s\le 1,\qquad0\le t\le 1\tag{21.3.6}$$

This parametric formulation is useful when the actual intersection point is required, for example in clipping or remapping.

For pure intersection classification, the orientation formulation is usually cleaner. Define,

$$o_1=\operatorname{orient2d}(a,b,c),\qquad o_2=\operatorname{orient2d}(a,b,d),\tag{21.3.7}$$

and,

$$o_3=\operatorname{orient2d}(c,d,a),\qquad o_4=\operatorname{orient2d}(c,d,b) \tag{21.3.8}$$

In the nondegenerate case, the two open segments properly cross if:

$$o_1 o_2 <0\qquad\text{and}\qquad o_3 o_4 <0 \tag{21.3.9}$$

This condition states that $c$ and $d$ lie on opposite sides of the oriented line through $a,b$, while $a$ and $b$ lie on opposite sides of the oriented line through $c,d$. The test uses only determinant signs and is therefore structurally compatible with the robust predicate framework of Section 21.2.

Boundary cases require explicit treatment. A point $p$ lies on the segment $[a,b]$ if:

$$\operatorname{orient2d}(a,b,p)=0 \tag{21.3.10}$$

and its coordinates lie between those of $a$ and $b$:

$$\min(a_x,b_x)\le p_x\le \max(a_x,b_x),\\ \min(a_y,b_y)\le p_y\le \max(a_y,b_y) \tag{21.3.11}$$

Thus, a complete segment-intersection predicate must distinguish at least four outcomes:

$$\text{Disjoint},\quad\text{Proper Crossing},\quad\text{Endpoint Contact},\quad\text{Collinear Overlap} \tag{21.3.12}$$

The last case occurs when,

$$\operatorname{orient2d}(a,b,c)=\operatorname{orient2d}(a,b,d)=0 \tag{21.3.13}$$

and the one-dimensional projections of the two segments overlap. If the segments are collinear, overlap can be tested along the dominant coordinate direction. For example, if $|b_x-a_x|\ge |b_y-a_y|$, the projected intervals overlap when,

$$\max(\min(a_x,b_x),\min(c_x,d_x))\le\min(\max(a_x,b_x),\max(c_x,d_x)) \tag{21.3.14}$$

A corresponding condition using the $y$-coordinates is used when the segment is more vertical.

The distinction between proper crossing, endpoint contact, and overlap is not cosmetic. In mesh intersection, endpoint contacts may preserve topology, while an unrecognized overlap may produce duplicate edges or nonmanifold cells. In conservative finite-volume remapping, a missed intersection changes the area or volume transferred between cells. In exact mesh-arrangement algorithms, many local segment or triangle intersections must be merged, sorted, and reconstructed consistently, which is why local correctness alone is insufficient unless it is embedded in a robust global policy (Guo and Fu, 2024; Lévy, 2025).

### Rust Implementation

Following the discussion in Section 21.3.1 on orientation predicates, affine segment parametrizations, and geometric classification logic, Program 21.3.1 presents a practical implementation of two-dimensional segment-segment intersection testing with explicit classification of all major geometric cases. The program combines determinant-based orientation predicates with affine line parametrization to distinguish proper crossings, endpoint contacts, collinear overlaps, and disjoint configurations in a numerically consistent manner. The implementation follows the mathematical structure introduced in equations (21.3.1)–(21.3.14), demonstrating how primitive predicates become higher-level geometric operations through carefully coordinated decision logic. Because segment intersection forms the foundation for clipping, polygon reconstruction, mesh intersection, cut-cell methods, and conservative remapping, the program emphasizes deterministic classification and explicit handling of degenerate configurations rather than relying on simplified boolean tests. The implementation also reflects the robustness concerns discussed throughout Sections 21.2 and 21.3, where small numerical inconsistencies may alter geometric topology or invalidate later reconstruction algorithms.

At the core of the implementation are the `Point2`, `Vector2`, and `Segment2` structures, which define the primitive geometric representation layer. The `Point2` structure stores planar coordinates and provides helper methods for vector construction, affine translation, and approximate equality checks. The `Vector2` structure encapsulates vector scaling operations used during affine parametrization of supporting lines. The `Segment2` structure stores segment endpoints and provides methods for computing direction vectors and affine segment evaluation. Together, these structures implement the coordinate and primitive-kernel layers described conceptually in equation (21.2.51), separating geometric representation from predicate evaluation and topological classification.

The enumeration `SegmentIntersection` encodes the four geometric outcomes introduced in equation (21.3.12): `Disjoint`, `ProperCrossing`, `EndpointContact`, and `CollinearOverlap`. Instead of returning a single boolean flag, the implementation preserves the complete geometric classification together with associated intersection geometry. Proper crossings and endpoint contacts return the corresponding intersection point, while collinear overlap returns the interval endpoints of the overlapping segment. This richer classification structure reflects the discussion in Section 21.3.1 that different intersection cases have distinct topological and numerical consequences in scientific computing applications.

The function `orient2d` implements the determinant predicate introduced in equations (21.3.7)–(21.3.8). This predicate computes the signed orientation of three points and forms the foundation of the segment-intersection classification logic. The sign of the determinant determines on which side of the oriented supporting line a query point lies. The implementation follows the determinant form of equation (21.2.10), thereby connecting the segment algorithm directly to the orientation predicates introduced earlier in Section 21.2.2.

The helper function `cross` evaluates the two-dimensional vector cross product used in the affine intersection formulation of equations (21.3.2)–(21.3.5). This function provides the determinant kernel required for evaluating line intersection parameters. The function `line_intersection_parameters` implements the affine-parametric intersection method derived from equations (21.3.2)–(21.3.6). Given two supporting lines, it computes the affine parameters $s$ and $t$ together with the corresponding geometric intersection point. If the determinant denominator vanishes within tolerance, the supporting lines are treated as parallel or nearly parallel and no unique affine intersection is returned. This function therefore provides the geometric intersection coordinates needed for clipping and remapping applications where the actual intersection point must be reconstructed explicitly.

The function `point_on_segment` implements the point-containment condition introduced in equations (21.3.10)–(21.3.11). The function first verifies collinearity using the orientation predicate and then checks whether the point coordinates lie within the bounding box of the segment endpoints. This explicit combination of orientation and interval logic is essential because collinearity alone does not imply membership in the finite segment. The helper function `in_range` provides the tolerance-aware interval comparisons used throughout the implementation.

The function `collinear_overlap` implements the overlap logic associated with equations (21.3.13)–(21.3.14). Once two segments have been classified as collinear, the function projects the segments onto their dominant coordinate direction and tests whether the projected intervals overlap. Depending on the interval structure, the function either returns a degenerate endpoint contact or a genuine overlapping segment interval. This explicit handling of collinear overlap is particularly important in mesh-processing and topology-reconstruction applications because unrecognized overlaps may create duplicate edges or inconsistent adjacency structures.

The central point of the implementation is the function `segment_intersection`. This function combines orientation predicates, affine line intersection, containment logic, and collinear overlap analysis into a complete segment-intersection classifier. The function first evaluates the orientation predicates from equations (21.3.7)–(21.3.8) and detects the special collinear case. If the segments are not collinear, the affine intersection parameters are computed and tested against the interval conditions of equation (21.3.6). Proper interior intersections are classified as `ProperCrossing`, while affine parameters near segment endpoints produce `EndpointContact`. If no valid affine intersection exists, the function falls back to explicit endpoint containment tests. This layered structure illustrates how primitive determinant signs are assembled into a robust higher-level geometric algorithm.

The helper function `print_intersection` converts the intersection classifications into readable diagnostic output. Rather than exposing raw numerical predicate values, the output presents geometrically meaningful classifications together with the associated intersection geometry. This mirrors the conceptual distinction emphasized throughout Section 21.3 between numerical predicate evaluation and topological interpretation.

The `main` function demonstrates the full range of segment-intersection configurations discussed in Section 21.3.1. The program evaluates proper crossings, endpoint contacts, collinear overlaps, collinear endpoint touches, parallel disjoint segments, and nonparallel disjoint segments. For the proper-crossing example, the program additionally prints the orientation values $o_1,o_2,o_3,o_4$ introduced in equations (21.3.7)–(21.3.9), verifying that the crossing condition is satisfied through opposite orientation signs. The affine intersection parameters $s$ and $t$ are also reported explicitly to demonstrate the connection between determinant-based classification and affine-parametric reconstruction. Finally, the program prints the tolerance-based classification policy used throughout the implementation, illustrating how geometric predicates are converted into deterministic algorithmic decisions in practical scientific-computing software.

```rust
// Program 21.3.1: Segment-Segment Intersection with Geometric Classification
//
// Problem statement:
// Implement a complete two-dimensional segment-intersection predicate for the
// cases discussed in Section 21.3.1. The program uses orientation tests,
// affine line parameters, endpoint containment tests, and collinear interval
// overlap to distinguish disjoint segments, proper crossings, endpoint
// contacts, and collinear overlaps.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Vector2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Segment2 {
    a: Point2,
    b: Point2,
}

#[derive(Clone, Copy, Debug)]
enum SegmentIntersection {
    Disjoint,
    ProperCrossing(Point2),
    EndpointContact(Point2),
    CollinearOverlap(Point2, Point2),
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn vector_to(self, other: Point2) -> Vector2 {
        Vector2 {
            x: other.x - self.x,
            y: other.y - self.y,
        }
    }

    fn add_vector(self, v: Vector2) -> Point2 {
        Point2 {
            x: self.x + v.x,
            y: self.y + v.y,
        }
    }

    fn approx_eq(self, other: Point2, tolerance: f64) -> bool {
        (self.x - other.x).abs() <= tolerance && (self.y - other.y).abs() <= tolerance
    }
}

impl Vector2 {
    fn scale(self, alpha: f64) -> Vector2 {
        Vector2 {
            x: alpha * self.x,
            y: alpha * self.y,
        }
    }
}

impl Segment2 {
    fn new(a: Point2, b: Point2) -> Self {
        Self { a, b }
    }

    fn direction(self) -> Vector2 {
        self.a.vector_to(self.b)
    }

    fn point_at(self, t: f64) -> Point2 {
        self.a.add_vector(self.direction().scale(t))
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
}

fn cross(u: Vector2, v: Vector2) -> f64 {
    u.x * v.y - u.y * v.x
}

fn in_range(value: f64, lower: f64, upper: f64, tolerance: f64) -> bool {
    value >= lower.min(upper) - tolerance && value <= lower.max(upper) + tolerance
}

fn point_on_segment(p: Point2, segment: Segment2, tolerance: f64) -> bool {
    orient2d(segment.a, segment.b, p).abs() <= tolerance
        && in_range(p.x, segment.a.x, segment.b.x, tolerance)
        && in_range(p.y, segment.a.y, segment.b.y, tolerance)
}

fn line_intersection_parameters(
    s1: Segment2,
    s2: Segment2,
    tolerance: f64,
) -> Option<(f64, f64, Point2)> {
    let r = s1.direction();
    let q_minus_p = s1.a.vector_to(s2.a);
    let s = s2.direction();

    let denominator = cross(r, s);

    if denominator.abs() <= tolerance {
        return None;
    }

    let alpha = cross(q_minus_p, s) / denominator;
    let beta = cross(q_minus_p, r) / denominator;

    let point = s1.point_at(alpha);

    Some((alpha, beta, point))
}

fn dominant_coordinate(p: Point2, q: Point2) -> usize {
    if (q.x - p.x).abs() >= (q.y - p.y).abs() {
        0
    } else {
        1
    }
}

fn coordinate(point: Point2, axis: usize) -> f64 {
    if axis == 0 {
        point.x
    } else {
        point.y
    }
}

fn collinear_overlap(s1: Segment2, s2: Segment2, tolerance: f64) -> SegmentIntersection {
    let axis = dominant_coordinate(s1.a, s1.b);

    let mut first = [
        (coordinate(s1.a, axis), s1.a),
        (coordinate(s1.b, axis), s1.b),
    ];

    let mut second = [
        (coordinate(s2.a, axis), s2.a),
        (coordinate(s2.b, axis), s2.b),
    ];

    first.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap());
    second.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap());

    let start_value = first[0].0.max(second[0].0);
    let end_value = first[1].0.min(second[1].0);

    if start_value > end_value + tolerance {
        return SegmentIntersection::Disjoint;
    }

    let start_point = if first[0].0 >= second[0].0 {
        first[0].1
    } else {
        second[0].1
    };

    let end_point = if first[1].0 <= second[1].0 {
        first[1].1
    } else {
        second[1].1
    };

    if start_point.approx_eq(end_point, tolerance) {
        SegmentIntersection::EndpointContact(start_point)
    } else {
        SegmentIntersection::CollinearOverlap(start_point, end_point)
    }
}

fn segment_intersection(
    s1: Segment2,
    s2: Segment2,
    tolerance: f64,
) -> SegmentIntersection {
    let o1 = orient2d(s1.a, s1.b, s2.a);
    let o2 = orient2d(s1.a, s1.b, s2.b);
    let o3 = orient2d(s2.a, s2.b, s1.a);
    let o4 = orient2d(s2.a, s2.b, s1.b);

    let s1_collinear_with_s2 = o1.abs() <= tolerance && o2.abs() <= tolerance;
    let s2_collinear_with_s1 = o3.abs() <= tolerance && o4.abs() <= tolerance;

    if s1_collinear_with_s2 && s2_collinear_with_s1 {
        return collinear_overlap(s1, s2, tolerance);
    }

    if let Some((alpha, beta, point)) = line_intersection_parameters(s1, s2, tolerance) {
        let alpha_inside = alpha >= -tolerance && alpha <= 1.0 + tolerance;
        let beta_inside = beta >= -tolerance && beta <= 1.0 + tolerance;

        if alpha_inside && beta_inside {
            let endpoint_contact =
                alpha.abs() <= tolerance
                    || (alpha - 1.0).abs() <= tolerance
                    || beta.abs() <= tolerance
                    || (beta - 1.0).abs() <= tolerance;

            if endpoint_contact {
                SegmentIntersection::EndpointContact(point)
            } else {
                SegmentIntersection::ProperCrossing(point)
            }
        } else {
            SegmentIntersection::Disjoint
        }
    } else {
        if point_on_segment(s2.a, s1, tolerance) {
            return SegmentIntersection::EndpointContact(s2.a);
        }

        if point_on_segment(s2.b, s1, tolerance) {
            return SegmentIntersection::EndpointContact(s2.b);
        }

        if point_on_segment(s1.a, s2, tolerance) {
            return SegmentIntersection::EndpointContact(s1.a);
        }

        if point_on_segment(s1.b, s2, tolerance) {
            return SegmentIntersection::EndpointContact(s1.b);
        }

        SegmentIntersection::Disjoint
    }
}

fn print_intersection(label: &str, result: SegmentIntersection) {
    match result {
        SegmentIntersection::Disjoint => {
            println!("{:>20} : Disjoint", label);
        }
        SegmentIntersection::ProperCrossing(p) => {
            println!(
                "{:>20} : ProperCrossing at ({:.8}, {:.8})",
                label, p.x, p.y
            );
        }
        SegmentIntersection::EndpointContact(p) => {
            println!(
                "{:>20} : EndpointContact at ({:.8}, {:.8})",
                label, p.x, p.y
            );
        }
        SegmentIntersection::CollinearOverlap(p0, p1) => {
            println!(
                "{:>20} : CollinearOverlap from ({:.8}, {:.8}) to ({:.8}, {:.8})",
                label, p0.x, p0.y, p1.x, p1.y
            );
        }
    }
}

fn main() {
    let tolerance = 1.0e-12;

    println!("Segment Intersection and Orientation Logic");
    println!("==========================================");
    println!();

    let cases = [
        (
            "proper crossing",
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 4.0)),
            Segment2::new(Point2::new(0.0, 4.0), Point2::new(4.0, 0.0)),
        ),
        (
            "endpoint contact",
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(2.0, 2.0)),
            Segment2::new(Point2::new(2.0, 2.0), Point2::new(4.0, 1.0)),
        ),
        (
            "collinear overlap",
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 0.0)),
            Segment2::new(Point2::new(2.0, 0.0), Point2::new(6.0, 0.0)),
        ),
        (
            "collinear touch",
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(2.0, 0.0)),
            Segment2::new(Point2::new(2.0, 0.0), Point2::new(4.0, 0.0)),
        ),
        (
            "parallel disjoint",
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(3.0, 0.0)),
            Segment2::new(Point2::new(0.0, 1.0), Point2::new(3.0, 1.0)),
        ),
        (
            "nonparallel disjoint",
            Segment2::new(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)),
            Segment2::new(Point2::new(2.0, 0.0), Point2::new(3.0, 1.0)),
        ),
    ];

    println!("Intersection Classifications");
    println!("----------------------------");

    for (label, s1, s2) in cases {
        let result = segment_intersection(s1, s2, tolerance);
        print_intersection(label, result);
    }

    println!();

    let diagnostic_s1 =
        Segment2::new(Point2::new(0.0, 0.0), Point2::new(4.0, 4.0));
    let diagnostic_s2 =
        Segment2::new(Point2::new(0.0, 4.0), Point2::new(4.0, 0.0));

    let o1 = orient2d(diagnostic_s1.a, diagnostic_s1.b, diagnostic_s2.a);
    let o2 = orient2d(diagnostic_s1.a, diagnostic_s1.b, diagnostic_s2.b);
    let o3 = orient2d(diagnostic_s2.a, diagnostic_s2.b, diagnostic_s1.a);
    let o4 = orient2d(diagnostic_s2.a, diagnostic_s2.b, diagnostic_s1.b);

    println!("Orientation Diagnostics for Proper Crossing Case");
    println!("------------------------------------------------");
    println!("o1 = orient2d(a,b,c) = {:.8}", o1);
    println!("o2 = orient2d(a,b,d) = {:.8}", o2);
    println!("o3 = orient2d(c,d,a) = {:.8}", o3);
    println!("o4 = orient2d(c,d,b) = {:.8}", o4);
    println!("o1 * o2              = {:.8}", o1 * o2);
    println!("o3 * o4              = {:.8}", o3 * o4);
    println!();

    if let Some((s, t, point)) =
        line_intersection_parameters(diagnostic_s1, diagnostic_s2, tolerance)
    {
        println!("Parametric Intersection Data");
        println!("----------------------------");
        println!("s parameter on first segment  = {:.8}", s);
        println!("t parameter on second segment = {:.8}", t);
        println!("intersection point            = ({:.8}, {:.8})", point.x, point.y);
    }

    println!();

    println!("Classification Policy");
    println!("---------------------");
    println!("Disjoint           : no intersection between the closed segments.");
    println!("ProperCrossing     : interiors of both segments intersect.");
    println!("EndpointContact    : intersection occurs at one or more endpoints.");
    println!("CollinearOverlap   : collinear segments share a nonzero-length interval.");
    println!("tolerance          = {:.1e}", tolerance);
}
```

Program 21.3.1 demonstrates how orientation predicates, affine coordinate systems, and interval tests combine to produce complete segment-intersection algorithms suitable for scientific computing applications. The implementation shows that reliable geometric algorithms require more than simply detecting whether an intersection exists. Different geometric cases must be classified explicitly because proper crossings, endpoint contacts, and collinear overlaps have fundamentally different topological consequences in mesh processing, clipping, remapping, and reconstruction algorithms.

The orientation-based formulation illustrates the structural simplicity of determinant predicates. Proper crossings are identified entirely through determinant signs, making the algorithm naturally compatible with robust predicate frameworks and exact arithmetic extensions. At the same time, the affine-parametric formulation demonstrates how actual geometric intersection points can be reconstructed when needed for conservative remapping, polygon clipping, or interface reconstruction.

The collinear-overlap logic further highlights the importance of degeneracy handling in computational geometry. Parallelism and collinearity are not exceptional corner cases but common geometric configurations in structured meshes, CAD-derived geometries, and aligned computational grids. Explicitly distinguishing overlap from simple endpoint contact prevents later topological inconsistencies and preserves deterministic geometric reconstruction.

The organization of the code also reflects the layered implementation philosophy discussed in Section 21.2.7. Primitive determinant kernels, affine coordinate calculations, containment predicates, and topological classifications are separated into distinct functions and data types. This modular structure allows higher-precision predicates, filtered arithmetic, or exact geometric kernels to be introduced later without requiring major modifications to the higher-level intersection logic.

## 21.3.2. Point-in-Polygon Tests and Boundary Conventions

A point-in-polygon algorithm classifies a query point $x\in\mathbb{R}^2$ relative to a polygon:

$$P=(p_0,p_1,\ldots,p_{n-1}),\qquad p_i\in\mathbb{R}^2 \tag{21.3.15}$$

with indices understood modulo $n$. The desired classification is:

$$x\in \operatorname{int}(P),\qquad x\in\partial P,\qquad x\notin P \tag{21.3.16}$$

For scientific computing, the boundary case must be explicit. A point lying exactly on a polygon edge may represent a mesh vertex, an interface crossing, a particle on a boundary, or a point shared by adjacent cells. Treating such cases as arbitrary inside/outside booleans can make later computations nondeterministic.

The simplest general method is the ray-crossing test. One casts a ray from $x$, usually in the positive $x$-direction, and counts how many polygon edges cross the ray. If the number of crossings is odd, $x$ is inside; if it is even, $x$ is outside:

$$x\in \operatorname{int}(P)\quad\Longleftrightarrow\quad N_{\mathrm{cross}}(x)\equiv 1 \pmod 2 \tag{21.3.17}$$

For the edge $[p_i,p_{i+1}]$, a standard crossing condition is:

$$(p_{i,y}>x_y)\ne(p_{i+1,y}>x_y) \tag{21.3.18}$$

together with,

$$x_x<p_{i,x}+\frac{(x_y-p_{i,y})(p_{i+1,x}-p_{i,x})}{p_{i+1,y}-p_{i,y}}  \tag{21.3.19}$$

The half-open convention in (21.3.18) prevents double-counting at vertices. Horizontal edges are excluded from crossing counts but must still be tested separately for boundary membership.

A mathematically richer alternative is the winding number. For a closed polygonal curve $P$, the winding number of $P$ around a point $x\notin \partial P$ is:

$$
w(x)
=
\frac{1}{2\pi}
\sum_{i=0}^{n-1}
\operatorname{atan2}
\left(
\det(p_i-x,p_{i+1}-x),
(p_i-x)\cdot(p_{i+1}-x)
\right)
\tag{21.3.20}
$$

For a simple positively oriented polygon,

$$x\in\operatorname{int}(P)\quad \Longleftrightarrow\quad w(x)=1 \tag{21.3.21}$$

and,

$$x\notin P\quad\Longleftrightarrow\quad w(x)=0 \tag{21.3.22}$$

For more general oriented curves, the winding number retains orientation information and can represent multiple coverings. This makes it conceptually useful in signed-distance calculations, polygonal Voronoi constructions, and geometric processing on oriented boundaries (Bálint, Bán and Valasek, 2024).

The boundary condition is again predicate-based. A point $x$ lies on an edge $[p_i,p_{i+1}]$ if:

$$\operatorname{orient2d}(p_i,p_{i+1},x)=0 \tag{21.3.23}$$

and

$$\min(p_{i,x},p_{i+1,x})\le x_x\le \max(p_{i,x},p_{i+1,x}) \tag{21.3.24}$$

$$\min(p_{i,y},p_{i+1,y})\le x_y\le \max(p_{i,y},p_{i+1,y}) \tag{21.3.25}$$

In robust implementations, this boundary test should be performed before the crossing or winding classification. Otherwise, a point on a vertex or edge may be counted differently depending on the chosen ray direction or floating-point perturbation.

For convex polygons, faster specialized tests are possible. If the vertices are sorted cyclically and the polygon is strictly convex, point location can be reduced to orientation tests against a fan triangulation or to binary search over angular sectors. Baseline algorithms are $O(n)$ per query, while ordered convex-polygon methods can achieve $O(\log n)$ query complexity after suitable preprocessing. Recent projective algorithms for point-in-convex-polygon and line-convex-polygon intersection emphasize that even classical containment tests remain active topics when robustness, degeneracy, and fast query behavior are considered together (Skala, 2025a; Skala, 2025b).

### Rust Implementation

Following the discussion in Section 21.3.2 on polygon containment, ray-crossing logic, winding numbers, and explicit boundary conventions, Program 21.3.2 presents a practical implementation of point-in-polygon classification for planar polygonal domains. The program combines orientation predicates, boundary tests, ray-crossing counts, and winding-number evaluation to distinguish interior, boundary, and exterior query points in a deterministic and numerically consistent manner. The implementation follows the mathematical structure introduced in equations (21.3.15)–(21.3.25), demonstrating how primitive orientation predicates become higher-level polygonal classification algorithms through carefully coordinated geometric decision logic. Because polygon containment tests arise in mesh traversal, conservative remapping, cut-cell methods, signed-distance evaluation, and interface reconstruction, the implementation emphasizes explicit treatment of boundary configurations rather than relying on simplified boolean inside/outside logic. The program also illustrates the importance of deterministic classification policies in scientific computing, where inconsistent handling of edge or vertex cases may propagate into nondeterministic mesh ownership, interpolation inconsistencies, or incorrect geometric reconstruction.

At the core of the implementation are the `Point2`, `Vector2`, and `Edge2` structures, which define the primitive geometric representation layer for polygonal computations. The `Point2` structure stores planar coordinates and provides methods for constructing displacement vectors between points. The `Vector2` structure implements dot products and determinant evaluations used in winding-number calculations and orientation logic. The `Edge2` structure stores polygon edges explicitly as ordered endpoint pairs. Together, these structures form the coordinate and primitive-kernel layers described conceptually in equation (21.2.51), separating geometric representation from predicate evaluation and topological classification.

The enumeration `PointPolygonClassification` encodes the three polygon-containment outcomes introduced in equation (21.3.16): `Inside`, `Outside`, and `Boundary`. Instead of collapsing all results into a boolean containment test, the implementation preserves explicit geometric classification. This distinction is essential in scientific computing because points on polygon boundaries frequently represent shared mesh vertices, interface intersections, or boundary particles whose treatment must remain deterministic and reproducible across adjacent geometric entities.

The function `orient2d` implements the determinant predicate introduced previously in equations (21.2.9)–(21.2.11). This orientation test is used throughout the implementation to detect collinearity between a query point and a polygon edge. The sign of the determinant determines whether a point lies to the left or right of an oriented edge, while vanishing determinant values indicate boundary alignment. The implementation therefore directly connects polygon classification to the orientation predicates introduced earlier in Section 21.2.2.

The helper functions `in_range` and `point_on_edge` implement the boundary conditions of equations (21.3.23)–(21.3.25). The `point_on_edge` function first verifies collinearity using the orientation predicate and then checks whether the query point lies within the coordinate ranges of the edge endpoints. This explicit combination of determinant and interval logic ensures that edge containment is treated separately from interior/exterior classification. The implementation follows the recommendation emphasized in Section 21.3.2 that boundary detection should occur before ray-crossing or winding-number evaluation, thereby avoiding inconsistent classification of points lying exactly on vertices or edges.

The function `polygon_edges` converts the polygon vertex list into an explicit collection of cyclic polygon edges. Since the polygon indices are interpreted modulo the number of vertices, the final edge automatically connects the last vertex back to the first. This explicit edge representation simplifies later traversal for boundary testing, ray crossing, and winding-number accumulation.

The function `boundary_test` applies the explicit edge-containment logic to every polygon edge. If the query point lies on any edge within the prescribed tolerance, the function immediately reports boundary membership. This boundary-first policy is one of the most important structural features of the implementation because it guarantees deterministic handling of edge and vertex cases before any parity or winding-number logic is applied.

The function `ray_crossing_classification` implements the ray-crossing algorithm described in equations (21.3.17)–(21.3.19). After first performing the boundary test, the function casts a conceptual ray from the query point in the positive $x$-direction and counts how many polygon edges intersect that ray. The half-open crossing convention of equation (21.3.18) is implemented through the logical comparison of vertex heights relative to the query point, preventing double-counting at polygon vertices. For each valid crossing edge, the corresponding intersection point along the ray is computed using the affine interpolation formula of equation (21.3.19). If the total number of crossings is odd, the point is classified as interior; otherwise, it is classified as exterior.

The functions `winding_number_value` and `winding_number_classification` implement the winding-number formulation introduced in equations (21.3.20)–(21.3.22). The `winding_number_value` function accumulates oriented turning angles around the query point using the determinant and dot-product formulation inside the `atan2` function. The resulting angle sum is normalized by $2\pi$, producing the winding number associated with the polygonal curve. The classification function then interprets nonzero winding numbers as interior points and zero winding numbers as exterior points. The implementation therefore illustrates the deeper geometric interpretation of polygon containment through oriented boundary traversal rather than parity counting alone.

The function `polygon_signed_area` evaluates the polygon signed area using the shoelace formula introduced earlier in equation (21.2.14). The sign of the computed area determines the orientation of the polygon vertex ordering, while the magnitude gives the polygonal area itself. This diagnostic function illustrates the close relationship between polygon orientation, winding conventions, and determinant-based geometry.

The helper function `print_polygon` outputs the polygon vertex list in readable form. This diagnostic output clarifies the cyclic polygon structure and makes the winding orientation visually interpretable. Although simple, such diagnostic utilities are often important in geometric software because topological errors are frequently easier to diagnose from explicit geometric output than from abstract numerical predicates alone.

The `main` function demonstrates the complete workflow of polygon classification. The program first constructs a simple polygon and computes its signed area to verify orientation consistency. It then evaluates representative query points corresponding to interior points, boundary-edge points, boundary-vertex points, and exterior points. For each query point, both the ray-crossing and winding-number classifiers are applied, and the resulting winding number is reported explicitly. The implementation thereby demonstrates agreement between the parity-based and orientation-based formulations of polygon containment. Finally, the program prints the boundary-first classification policy used throughout the implementation, emphasizing the importance of deterministic treatment of edge and vertex cases in scientific-computing applications.

```rust
// Program 21.3.2: Point-in-Polygon Classification with Boundary Conventions
//
// Problem statement:
// Implement point-in-polygon classification for a two-dimensional polygon.
// The program distinguishes Inside, Outside, and Boundary cases. It first
// applies an explicit boundary test based on orientation and coordinate ranges,
// then applies both the ray-crossing test and the winding-number test for
// interior/exterior classification.

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Vector2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Edge2 {
    a: Point2,
    b: Point2,
}

#[derive(Clone, Copy, Debug)]
enum PointPolygonClassification {
    Inside,
    Outside,
    Boundary,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn vector_to(self, other: Point2) -> Vector2 {
        Vector2 {
            x: other.x - self.x,
            y: other.y - self.y,
        }
    }
}

impl Vector2 {
    fn dot(self, other: Vector2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn det(self, other: Vector2) -> f64 {
        self.x * other.y - self.y * other.x
    }
}

impl Edge2 {
    fn new(a: Point2, b: Point2) -> Self {
        Self { a, b }
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
}

fn in_range(value: f64, lower: f64, upper: f64, tolerance: f64) -> bool {
    value >= lower.min(upper) - tolerance && value <= lower.max(upper) + tolerance
}

fn point_on_edge(x: Point2, edge: Edge2, tolerance: f64) -> bool {
    orient2d(edge.a, edge.b, x).abs() <= tolerance
        && in_range(x.x, edge.a.x, edge.b.x, tolerance)
        && in_range(x.y, edge.a.y, edge.b.y, tolerance)
}

fn polygon_edges(polygon: &[Point2]) -> Vec<Edge2> {
    let n = polygon.len();

    if n < 2 {
        return Vec::new();
    }

    let mut edges = Vec::with_capacity(n);

    for i in 0..n {
        edges.push(Edge2::new(polygon[i], polygon[(i + 1) % n]));
    }

    edges
}

fn boundary_test(polygon: &[Point2], x: Point2, tolerance: f64) -> bool {
    for edge in polygon_edges(polygon) {
        if point_on_edge(x, edge, tolerance) {
            return true;
        }
    }

    false
}

fn ray_crossing_classification(
    polygon: &[Point2],
    x: Point2,
    tolerance: f64,
) -> PointPolygonClassification {
    if polygon.len() < 3 {
        return PointPolygonClassification::Outside;
    }

    if boundary_test(polygon, x, tolerance) {
        return PointPolygonClassification::Boundary;
    }

    let mut crossings = 0usize;

    for edge in polygon_edges(polygon) {
        let yi_above = edge.a.y > x.y;
        let yj_above = edge.b.y > x.y;

        if yi_above != yj_above {
            let x_intersection = edge.a.x
                + (x.y - edge.a.y) * (edge.b.x - edge.a.x)
                    / (edge.b.y - edge.a.y);

            if x.x < x_intersection {
                crossings += 1;
            }
        }
    }

    if crossings % 2 == 1 {
        PointPolygonClassification::Inside
    } else {
        PointPolygonClassification::Outside
    }
}

fn winding_number_value(polygon: &[Point2], x: Point2) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }

    let mut angle_sum = 0.0;

    for edge in polygon_edges(polygon) {
        let u = x.vector_to(edge.a);
        let v = x.vector_to(edge.b);

        let det = u.det(v);
        let dot = u.dot(v);

        angle_sum += det.atan2(dot);
    }

    angle_sum / (2.0 * PI)
}

fn winding_number_classification(
    polygon: &[Point2],
    x: Point2,
    tolerance: f64,
) -> PointPolygonClassification {
    if polygon.len() < 3 {
        return PointPolygonClassification::Outside;
    }

    if boundary_test(polygon, x, tolerance) {
        return PointPolygonClassification::Boundary;
    }

    let winding = winding_number_value(polygon, x);

    if winding.abs() > 0.5 {
        PointPolygonClassification::Inside
    } else {
        PointPolygonClassification::Outside
    }
}

fn polygon_signed_area(polygon: &[Point2]) -> f64 {
    let n = polygon.len();

    if n < 3 {
        return 0.0;
    }

    let mut sum = 0.0;

    for i in 0..n {
        let p = polygon[i];
        let q = polygon[(i + 1) % n];

        sum += p.x * q.y - q.x * p.y;
    }

    0.5 * sum
}

fn print_polygon(polygon: &[Point2]) {
    for (i, p) in polygon.iter().enumerate() {
        println!("p_{:<2} = ({:>8.4}, {:>8.4})", i, p.x, p.y);
    }
}

fn main() {
    let tolerance = 1.0e-12;

    let polygon = vec![
        Point2::new(0.0, 0.0),
        Point2::new(4.0, 0.0),
        Point2::new(4.0, 2.0),
        Point2::new(2.0, 3.0),
        Point2::new(0.0, 2.0),
    ];

    let query_points = [
        ("inside", Point2::new(2.0, 1.0)),
        ("boundary edge", Point2::new(4.0, 1.0)),
        ("boundary vertex", Point2::new(2.0, 3.0)),
        ("outside", Point2::new(5.0, 1.0)),
        ("near top", Point2::new(2.0, 2.5)),
    ];

    println!("Point-in-Polygon Tests and Boundary Conventions");
    println!("================================================");
    println!();

    println!("Polygon Vertices");
    println!("----------------");
    print_polygon(&polygon);
    println!();

    println!("Polygon Diagnostics");
    println!("-------------------");
    println!("number of vertices        = {}", polygon.len());
    println!("signed area               = {:.10}", polygon_signed_area(&polygon));
    println!("tolerance                 = {:.1e}", tolerance);
    println!();

    println!("Point Classifications");
    println!("---------------------");
    println!(
        "{:>16} {:>20} {:>18} {:>18} {:>18}",
        "case", "query point", "ray crossing", "winding", "winding value"
    );

    for (label, x) in query_points {
        let ray_class = ray_crossing_classification(&polygon, x, tolerance);
        let winding_class = winding_number_classification(&polygon, x, tolerance);
        let winding = if boundary_test(&polygon, x, tolerance) {
            0.0
        } else {
            winding_number_value(&polygon, x)
        };

        println!(
            "{:>16} ({:>8.4}, {:>8.4}) {:>18?} {:>18?} {:>18.8}",
            label, x.x, x.y, ray_class, winding_class, winding
        );
    }

    println!();

    println!("Boundary-First Convention");
    println!("-------------------------");
    println!("1. Test whether the query point lies on any polygon edge.");
    println!("2. If it lies on an edge or vertex, return Boundary immediately.");
    println!("3. Otherwise, apply ray crossing or winding-number classification.");
}
```

Program 21.3.2 demonstrates how polygon containment algorithms are built from primitive orientation predicates, interval tests, and consistent boundary conventions. The implementation shows that reliable point-in-polygon classification requires more than simply counting ray crossings or evaluating winding numbers. Boundary points must be handled explicitly and deterministically before interior/exterior logic is applied. This reflects the broader computational-geometry principle emphasized throughout Sections 21.2 and 21.3: topological consistency depends critically on the coordination of local geometric decisions.

The ray-crossing implementation illustrates the computational simplicity of parity-based containment tests. Using only affine interpolation and parity counting, the algorithm provides an efficient $O(n)$ method for polygon containment. At the same time, the winding-number implementation reveals the richer geometric interpretation of polygon orientation and multiple coverings. While both methods agree for simple polygons, the winding-number formulation naturally generalizes to more complicated oriented curves and signed geometric constructions.

The explicit boundary-first logic further demonstrates the importance of robust classification conventions. Without a separate edge-containment test, points lying on polygon boundaries may be classified inconsistently depending on ray direction, vertex ordering, or floating-point perturbation. Such inconsistencies may later propagate into nondeterministic mesh traversal, incorrect interpolation ownership, or inconsistent interface reconstruction.

The organization of the implementation also reflects the layered design philosophy discussed in Section 21.2.7. Primitive determinant kernels, boundary predicates, parity logic, winding-number evaluation, and topological classifications are separated into distinct functions and data types. This modular structure allows filtered predicates, exact arithmetic kernels, or accelerated convex-polygon search algorithms to be incorporated later without rewriting the higher-level containment framework.

## 21.3.3. Line, Half-Space, and Polygon Clipping

Clipping algorithms compute the part of a geometric object that lies inside a prescribed region. In two dimensions, the most common operation is clipping a polygon by a half-plane. Let the oriented line through $a,b\in\mathbb{R}^2$ define the closed left half-plane:

$$
H^+(a,b)
=
\{x\in\mathbb{R}^{2}:\operatorname{orient2d}(a,b,x)\ge 0\}
\tag{21.3.26}
$$

Given a polygon $P$, the clipped polygon is:

$$P' = P\cap H^+(a,b) \tag{21.3.27}$$

A standard clipping algorithm processes each polygon edge $[p_i,p_{i+1}]$, classifies its endpoints relative to the half-plane, and emits zero, one, or two vertices depending on whether the edge remains inside, exits, enters, or stays outside.

For an edge from $u$ to $v$, define:

$$\phi(u)=\operatorname{orient2d}(a,b,u),\qquad\phi(v)=\operatorname{orient2d}(a,b,v) \tag{21.3.28}$$

The endpoint $u$ is inside if $\phi(u)\ge 0$. If the edge crosses the clipping line, the intersection point has the affine form:

$$z = u+t(v-u) \tag{21.3.29}$$

where,

$$t=\frac{\phi(u)}{\phi(u)-\phi(v)} \tag{21.3.30}$$

This follows from the linearity of the orientation determinant along the segment:

$$
\phi(u+t(v-u))
=
(1-t)\phi(u)+t\phi(v)
\tag{21.3.31}
$$

The edge crosses the clipping boundary when $\phi(u)$ and $\phi(v)$ have opposite signs. Boundary cases occur when either value is zero.

Repeated half-plane clipping can compute the intersection of a polygon with a convex polygon. If,

$$Q=\bigcap_{j=0}^{m-1} H_j\tag{21.3.32}$$

is a convex polygon represented as an intersection of oriented half-planes, then,

$$
P\cap Q
=
(((P\cap H_0)\cap H_1)\cdots)\cap H_{m-1}
\tag{21.3.33}
$$

In the baseline algorithm, clipping an $n$-vertex polygon by one half-plane is $O(n)$. Clipping by an $m$-edge convex polygon is therefore $O(mn)$ in the simplest implementation, although specialized algorithms can improve this for convex inputs and repeated queries.

In scientific computing, clipping is most important when the clipped geometry carries measure. For a finite-volume cell $K$ and a material region $D$, one often needs the area or volume fraction,

$$\alpha_K = \frac{|K\cap D|}{|K|} \tag{21.3.34}$$

In volume-of-fluid methods, cut-cell methods, and conservative remapping, such fractions directly affect conservation. The computation is no longer a visual operation; it becomes part of the discretized physical model. In higher dimensions, the corresponding operation is polytope intersection by half-spaces and hyperplanes. Recent work on polytope intersection for unsplit geometric volume-of-fluid methods demonstrates that the speed and accuracy of such geometric clipping operations are central to arbitrary-grid finite-volume computations (López and Hernández, 2024).

For a convex polytope in $\mathbb{R}^d$, a half-space can be written as:

$$H=\{x\in\mathbb{R}^d:n\cdot x\le \beta\} \tag{21.3.35}$$

Clipping a cell by $H$ requires classifying vertices using the signed quantity,

$$\psi(x)=n\cdot x-\beta \tag{21.3.36}$$

An edge $[u,v]$ crosses the clipping hyperplane when $\psi(u)\psi(v)<0$. The intersection point is:

$$z=u+t(v-u),\qquad t=\frac{\psi(u)}{\psi(u)-\psi(v)} \tag{21.3.37}$$

Thus, the same affine-interpolation structure used for planar clipping extends to arbitrary dimension. The main additional difficulty is topological reconstruction: after vertices and edges are cut, new faces and cell adjacencies must be generated consistently.

### Rust Implementation

Following the discussion in Section 21.3.3 on half-plane clipping, affine edge interpolation, and polygon intersection, Program 21.3.3 presents a practical implementation of polygon clipping against oriented half-planes and convex clipping regions. The program implements the classical edge-by-edge clipping logic described in equations (21.3.26)–(21.3.37), demonstrating how determinant predicates, affine interpolation, and geometric classification combine to produce robust polygon-intersection operations. The implementation illustrates both single half-plane clipping and repeated convex-polygon clipping, while also computing geometric area fractions relevant to finite-volume and cut-cell methods. Because clipping operations appear directly in conservative remapping, interface reconstruction, embedded-boundary discretization, and volume-of-fluid methods, the program emphasizes explicit handling of inside/outside transitions and numerically stable edge-intersection construction. The resulting implementation demonstrates how local geometric predicates become measure-preserving geometric operators within scientific-computing workflows.

At the core of the implementation are the `Point2` and `Line2` structures, which define the primitive geometric representation layer used throughout the clipping algorithms. The `Point2` structure stores planar coordinates and provides the `lerp` method for affine interpolation between vertices. This interpolation directly implements the segment representation introduced in equation (21.3.29), where an intersection point on an edge is written in affine form as $z=u+t(v-u)$. The `Line2` structure stores the oriented clipping line defining the half-plane $H^+(a,b)$ of equation (21.3.26). Together, these structures separate coordinate representation from clipping logic, consistent with the layered geometric architecture introduced earlier in equation (21.2.51).

The function `orient2d` implements the determinant predicate used throughout the clipping process. This predicate evaluates the signed orientation of a query point relative to the oriented clipping line. The helper function `signed_distance_to_half_plane` converts this determinant into the scalar quantity $\phi(x)$ defined in equation (21.3.28). The sign of this value determines whether a point lies inside or outside the clipping half-plane. The function `is_inside_half_plane` then implements the classification condition $\phi(x)\ge 0$, while also incorporating a floating-point tolerance to stabilize near-boundary classifications.

The function `edge_line_intersection` computes the affine intersection point between a polygon edge and the clipping boundary. Using the endpoint predicate values $\phi(u)$ and $\phi(v)$, the function evaluates the affine parameter $t$ from equation (21.3.30). The final intersection point is then computed through linear interpolation using the `lerp` method. This implementation directly reflects the linearity property described in equation (21.3.31), where the orientation determinant varies linearly along an edge segment. The function therefore demonstrates how determinant predicates and affine interpolation combine to produce stable clipping intersections.

The central clipping routine is implemented in the function `clip_polygon_by_half_plane`. This function processes every polygon edge sequentially and classifies the edge transition into one of four geometric cases: inside-to-inside, inside-to-outside, outside-to-inside, or outside-to-outside. Depending on the transition type, the function emits zero, one, or two output vertices. This logic is precisely the geometric classification process described following equation (21.3.27). Inside-to-inside edges preserve the endpoint, inside-to-outside edges emit only the clipping intersection, outside-to-inside edges emit both the intersection and the endpoint, and outside-to-outside edges contribute nothing. The resulting algorithm is a direct implementation of the classical Sutherland–Hodgman polygon clipping strategy.

The function `clip_polygon_by_convex_polygon` implements repeated half-plane clipping according to equations (21.3.32) and (21.3.33). The convex clipping polygon is interpreted as the intersection of multiple oriented half-planes, and the input polygon is clipped sequentially against each clipping edge. After each clipping stage, the intermediate polygon becomes the input for the next half-plane. This iterative structure demonstrates how complex convex-polygon intersection problems can be decomposed into repeated local clipping operations.

The functions `polygon_signed_area` and `polygon_area` compute polygon measure using the shoelace formula introduced previously in equation (21.2.14). The signed area preserves orientation information, while the absolute value gives the geometric area itself. These functions are important because clipping operations in scientific computing frequently carry physical meaning through area or volume fractions. The program therefore computes the finite-volume area fraction $\alpha_K$ defined in equation (21.3.34), illustrating how clipped geometry directly influences conservative numerical discretizations.

The helper function `remove_consecutive_duplicates` performs geometric cleanup after clipping. Because clipping operations may generate repeated vertices along edges or boundaries, the function removes nearly identical consecutive points using a tolerance-based comparison. This step improves numerical stability and prevents degenerate polygon structures from propagating into later geometric calculations. The companion function `points_close` provides the tolerance-based coordinate comparison used throughout the cleanup stage.

The utility function `print_polygon` outputs polygon vertices together with signed area and total area diagnostics. This function is especially useful in geometric algorithms because orientation and topology errors are often easier to diagnose from explicit polygon output than from abstract determinant values alone. The output also makes the effects of clipping geometrically transparent by displaying the transformed polygonal boundaries directly.

The `main` function demonstrates the complete clipping workflow. The program first constructs a rectangular polygon and clips it against a single oriented half-plane, illustrating equations (21.3.26)–(21.3.31). It then constructs a convex clipping polygon and applies repeated half-plane clipping according to equations (21.3.32) and (21.3.33). Finally, the program computes the clipped area fraction $\alpha_K$ from equation (21.3.34), demonstrating how geometric clipping becomes part of conservative finite-volume calculations. The final diagnostic output summarizes the clipping policy used for each edge-transition configuration and reports the floating-point tolerance used throughout the implementation.

```rust
// Program 21.3.3: Half-Plane Polygon Clipping and Area Fraction Computation
//
// Problem statement:
// Implement polygon clipping by an oriented half-plane in two dimensions.
// The program clips a polygon against the closed left half-plane defined by
// an oriented line, repeatedly clips a polygon by a convex clipping polygon,
// and computes the clipped area fraction used in finite-volume and cut-cell
// applications.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Line2 {
    a: Point2,
    b: Point2,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn lerp(self, other: Point2, t: f64) -> Point2 {
        Point2 {
            x: self.x + t * (other.x - self.x),
            y: self.y + t * (other.y - self.y),
        }
    }
}

impl Line2 {
    fn new(a: Point2, b: Point2) -> Self {
        Self { a, b }
    }
}

fn orient2d(a: Point2, b: Point2, x: Point2) -> f64 {
    (a.x - x.x) * (b.y - x.y) - (a.y - x.y) * (b.x - x.x)
}

fn signed_distance_to_half_plane(line: Line2, x: Point2) -> f64 {
    orient2d(line.a, line.b, x)
}

fn is_inside_half_plane(line: Line2, x: Point2, tolerance: f64) -> bool {
    signed_distance_to_half_plane(line, x) >= -tolerance
}

fn edge_line_intersection(line: Line2, u: Point2, v: Point2) -> Point2 {
    let phi_u = signed_distance_to_half_plane(line, u);
    let phi_v = signed_distance_to_half_plane(line, v);

    let denominator = phi_u - phi_v;

    if denominator.abs() <= 1.0e-14 {
        panic!("Cannot compute a stable clipping intersection for this edge.");
    }

    let t = phi_u / denominator;
    u.lerp(v, t)
}

fn clip_polygon_by_half_plane(
    polygon: &[Point2],
    line: Line2,
    tolerance: f64,
) -> Vec<Point2> {
    if polygon.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::new();
    let n = polygon.len();

    for i in 0..n {
        let u = polygon[i];
        let v = polygon[(i + 1) % n];

        let u_inside = is_inside_half_plane(line, u, tolerance);
        let v_inside = is_inside_half_plane(line, v, tolerance);

        match (u_inside, v_inside) {
            (true, true) => {
                output.push(v);
            }
            (true, false) => {
                let z = edge_line_intersection(line, u, v);
                output.push(z);
            }
            (false, true) => {
                let z = edge_line_intersection(line, u, v);
                output.push(z);
                output.push(v);
            }
            (false, false) => {}
        }
    }

    remove_consecutive_duplicates(&output, tolerance)
}

fn clip_polygon_by_convex_polygon(
    polygon: &[Point2],
    clipper: &[Point2],
    tolerance: f64,
) -> Vec<Point2> {
    let mut current = polygon.to_vec();

    if current.is_empty() || clipper.len() < 3 {
        return Vec::new();
    }

    for i in 0..clipper.len() {
        let a = clipper[i];
        let b = clipper[(i + 1) % clipper.len()];
        let line = Line2::new(a, b);

        current = clip_polygon_by_half_plane(&current, line, tolerance);

        if current.is_empty() {
            break;
        }
    }

    current
}

fn polygon_signed_area(polygon: &[Point2]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }

    let mut sum = 0.0;

    for i in 0..polygon.len() {
        let p = polygon[i];
        let q = polygon[(i + 1) % polygon.len()];

        sum += p.x * q.y - q.x * p.y;
    }

    0.5 * sum
}

fn polygon_area(polygon: &[Point2]) -> f64 {
    polygon_signed_area(polygon).abs()
}

fn remove_consecutive_duplicates(points: &[Point2], tolerance: f64) -> Vec<Point2> {
    let mut cleaned: Vec<Point2> = Vec::new();

    for &p in points {
        if cleaned
            .last()
            .map_or(true, |last| !points_close(*last, p, tolerance))
        {
            cleaned.push(p);
        }
    }

    if cleaned.len() > 1 {
        let first = cleaned[0];
        let last = *cleaned.last().unwrap();

        if points_close(first, last, tolerance) {
            cleaned.pop();
        }
    }

    cleaned
}

fn points_close(a: Point2, b: Point2, tolerance: f64) -> bool {
    (a.x - b.x).abs() <= tolerance && (a.y - b.y).abs() <= tolerance
}

fn print_polygon(label: &str, polygon: &[Point2]) {
    println!("{}", label);
    println!("{}", "-".repeat(label.len()));

    if polygon.is_empty() {
        println!("empty polygon");
        println!();
        return;
    }

    for (i, p) in polygon.iter().enumerate() {
        println!("p_{:<2} = ({:>10.6}, {:>10.6})", i, p.x, p.y);
    }

    println!("number of vertices = {}", polygon.len());
    println!("signed area        = {:.10}", polygon_signed_area(polygon));
    println!("area               = {:.10}", polygon_area(polygon));
    println!();
}

fn main() {
    let tolerance = 1.0e-12;

    println!("Line, Half-Space, and Polygon Clipping");
    println!("======================================");
    println!();

    let polygon = vec![
        Point2::new(0.0, 0.0),
        Point2::new(4.0, 0.0),
        Point2::new(4.0, 3.0),
        Point2::new(0.0, 3.0),
    ];

    let half_plane_line = Line2::new(Point2::new(2.0, -1.0), Point2::new(2.0, 4.0));
    let clipped_half_plane = clip_polygon_by_half_plane(&polygon, half_plane_line, tolerance);

    print_polygon("Original Polygon", &polygon);

    println!("Half-Plane Definition");
    println!("---------------------");
    println!(
        "oriented line a -> b = ({:.4}, {:.4}) -> ({:.4}, {:.4})",
        half_plane_line.a.x, half_plane_line.a.y, half_plane_line.b.x, half_plane_line.b.y
    );
    println!("inside condition      = orient2d(a,b,x) >= 0");
    println!();

    print_polygon("Polygon Clipped by One Half-Plane", &clipped_half_plane);

    let clipper = vec![
        Point2::new(1.0, 0.5),
        Point2::new(3.5, 0.5),
        Point2::new(3.5, 2.5),
        Point2::new(1.0, 2.5),
    ];

    let clipped_convex = clip_polygon_by_convex_polygon(&polygon, &clipper, tolerance);

    print_polygon("Convex Clipping Polygon", &clipper);
    print_polygon("Polygon Clipped by Convex Polygon", &clipped_convex);

    let original_area = polygon_area(&polygon);
    let clipped_area = polygon_area(&clipped_convex);
    let area_fraction = if original_area > 0.0 {
        clipped_area / original_area
    } else {
        0.0
    };

    println!("Finite-Volume Area Fraction");
    println!("---------------------------");
    println!("original cell area          = {:.10}", original_area);
    println!("clipped cell area           = {:.10}", clipped_area);
    println!("area fraction alpha_K       = {:.10}", area_fraction);
    println!();

    println!("Clipping Policy");
    println!("---------------");
    println!("Inside -> Inside    : emit the edge endpoint.");
    println!("Inside -> Outside   : emit the clipping intersection.");
    println!("Outside -> Inside   : emit the intersection, then the endpoint.");
    println!("Outside -> Outside  : emit nothing.");
    println!("tolerance            = {:.1e}", tolerance);
}
```

Program 21.3.3 demonstrates how determinant predicates, affine interpolation, and geometric classification combine to form robust polygon-clipping algorithms. The implementation shows that clipping is fundamentally a local edge-classification process driven by the signs of orientation predicates. By processing each polygon edge independently and reconstructing the resulting clipped polygon incrementally, the algorithm transforms primitive geometric decisions into higher-level polygon-intersection operations.

The repeated convex-polygon clipping procedure further illustrates how complex geometric intersections can be decomposed into sequential half-plane operations. This layered structure is especially important in scientific computing because many finite-volume, cut-cell, and interface-reconstruction methods require repeated clipping against multiple geometric constraints. The implementation therefore reflects the broader computational-geometry principle that local predicate evaluations must combine consistently to preserve global topological structure.

The area-fraction computation demonstrates that clipping operations in scientific computing are not merely graphical procedures. The clipped geometry carries physical measure, directly affecting conservative transfer operators, embedded-boundary fluxes, and material-volume reconstruction. The computed fraction $\alpha_K$ therefore becomes part of the discretized numerical model itself rather than simply a visualization artifact.

The organization of the implementation also follows the separation-of-concerns philosophy described in Section 21.2.7. Primitive determinant evaluations, half-plane classifications, affine intersection construction, polygon reconstruction, and measure evaluation are separated into modular functions and data structures. This structure allows exact predicates, filtered arithmetic, higher-dimensional clipping kernels, or advanced polytope-reconstruction methods to be incorporated later without modifying the higher-level clipping framework.

## 21.3.4. Polygon and Mesh Intersections in Simulation Pipelines

Local segment and polygon intersection formulas are mathematically simple, but simulation pipelines require global consistency. A mesh intersection algorithm may need to intersect thousands or millions of triangles, polygons, or polyhedra. Each local intersection produces points, edges, or faces that must be sorted, merged, deduplicated, and inserted into a consistent arrangement. The global problem is therefore not only to compute:

$$A_i\cap B_j \tag{21.3.38}$$

for individual geometric entities $A_i$ and $B_j$, but to construct a coherent cell complex representing all intersections simultaneously.

For two polygonal or polyhedral decompositions,

$$
\mathcal{A}=\{A_i\}_{i=1}^{N_A},
\qquad
\mathcal{B}=\{B_j\}_{j=1}^{N_B}
\tag{21.3.39}
$$

the overlay or arrangement consists of all nonempty intersections:

$$C_{ij}=A_i\cap B_j,\qquad C_{ij}\ne\varnothing \tag{21.3.40}$$

together with their induced vertices, edges, faces, and adjacency relations. In conservative transfer between meshes, quantities may be integrated over these intersection cells:

$$
\int_{A_i}u(x)\,dx
\approx
\sum_{j:C_{ij}\ne\varnothing}
\int_{C_{ij}}u(x)\,dx
\tag{21.3.41}
$$

If the geometry of $C_{ij}$ is inconsistent, conservation can fail even if the quadrature and finite-volume formulas are otherwise correct.

Robust mesh arrangements require more than accurate local intersection coordinates. They also require consistent symbolic identities. If two intersection points are mathematically identical, the algorithm must represent them as the same vertex. If an edge is split by multiple intersections, the split order must be consistent. If several triangles meet near a common line or point, local decisions must not create gaps, overlaps, or nonmanifold configurations. This is why exact intersection resolution and exact mesh CSG methods emphasize combinatorial reconstruction in addition to predicate correctness (Guo and Fu, 2024; Lévy, 2025).

The same issue appears in constructive solid geometry. Suppose a solid is represented through Boolean operations,

$$D=(D_1\cup D_2)\setminus D_3 \tag{21.3.42}$$

Point containment, boundary extraction, and mesh generation require repeated classification of points and surface elements relative to the participating solids. If one local test classifies a point as inside and another related test classifies it as outside because of finite-precision inconsistency, the Boolean result may contain cracks or invalid topology. Modern point-containment and exact CSG algorithms therefore treat classification, traversal, and reconstruction as coupled problems rather than independent subroutines (Romano et al., 2025; Lévy, 2025).

Spherical and curved geometries add another layer of difficulty. On the sphere, intersections are not governed by planar straight segments but by great-circle or curved arcs. Nevertheless, the same principles remain: geometric quantities must be represented consistently, intersections must be classified deterministically, and the resulting cells must preserve conservation and topology. Recent spherical intersection algorithms are especially relevant for geophysical remapping and climate-related discretizations, where geometric errors can translate directly into conservation errors on the sphere (Chen, Ullrich and Panetta, 2026).

## 21.3.5. Complexity, Robustness, and Implementation Structure

The elementary operations in this section have simple baseline costs. Segment-segment intersection is $O(1)$. Point-in-simple-polygon testing by ray crossing or winding number is $O(n)$ for an $n$-vertex polygon. Point-in-convex-polygon testing is $O(n)$ by direct half-plane checks and can be reduced to $O(\log n)$ with ordered vertices and suitable preprocessing. Polygon clipping by one half-plane is $O(n)$, while repeated clipping by an $m$-edge convex polygon has a straightforward $O(mn)$ baseline cost. These costs are important, but in scientific computing the asymptotic count is only part of the story. Robustness, memory layout, degeneracy handling, and topology reconstruction often dominate practical reliability.

A complete implementation should therefore avoid representing geometric decisions as booleans whenever more structure is present. Segment intersection should return a classification such as:

$$
\text{None},\quad
\text{Point}(z),\quad
\text{Overlap}([z_0,z_1])
\tag{21.3.43}
$$

rather than only true or false. Point-in-polygon should return,

$$\text{Inside},\quad\text{Outside},\quad\text{Boundary} \tag{21.3.44}$$

Clipping should preserve provenance information when needed, indicating whether a vertex is original or generated by intersection. Mesh arrangement code should maintain stable identifiers for vertices, edges, faces, and cells so that duplicate geometric entities can be merged consistently.

From a Rust perspective, this suggests a layered design:

$$

\text{predicate kernel}
\longrightarrow
\text{classification enum}
\longrightarrow
\text{local intersection object}\\
\longrightarrow
\text{topological reconstruction}
\tag{21.3.45}
$$

The predicate kernel evaluates orientation, sidedness, and affine parameters. The classification layer records discrete outcomes. The local intersection layer computes points, overlaps, and clipped polygons. The reconstruction layer updates connectivity and adjacency. Keeping these layers separate prevents low-level floating-point choices from leaking silently into mesh topology.

The main practical warning is that tolerances should not be scattered throughout the code as unrelated constants. If a tolerance policy is unavoidable, it should be localized, documented, and applied consistently. For production-quality geometric algorithms, especially those that affect mesh topology or conservation, filtered or exact predicates are preferable. Segment, polygon, and clipping algorithms are mathematically elementary, but their role in scientific computing is structurally critical: they are the bridge between determinant predicates and the reliable geometric transformations needed by numerical methods.

# 21.4. Convex Hulls

Convex hulls are among the most classical objects in computational geometry, but their role in scientific computing is broader than their elementary definition might suggest. They provide outer approximations of point sets, support fast rejection tests, define bounding regions, assist in collision and contact preprocessing, and serve as a conceptual bridge between point sets, half-space intersections, triangulations, and dual geometric constructions. In mesh generation and point-cloud analysis, convex hulls often appear as a first structural summary of sampled data. In simulation pipelines, they are also useful for detecting extreme points, building coarse geometric envelopes, and initializing more refined spatial data structures. Although convex hull algorithms are mathematically mature, recent implementation-oriented work still emphasizes robustness, degeneracy handling, dynamic updates, and the difference between formula-level correctness and reliable software behavior (Gæde et al., 2024; Kwon, Oh and Baek, 2024).

## 21.4.1. Convex Sets, Extreme Points, and Supporting Hyperplanes

A set $C\subset \mathbb{R}^d$ is convex if, for any two points $x,y\in C$, the full segment between them lies in $C:$

$$(1-t)x+ty\in C,\qquad 0\le t\le 1\tag{21.4.1}$$

Given a finite point set,

$$P={p_1,p_2,\ldots,p_n}\subset \mathbb{R}^d \tag{21.4.2}$$

its convex hull is the smallest convex set containing all points in $P$. Equivalently, it is the set of all convex combinations of the points:

$$
\operatorname{conv}(P)
=
\left\{
\sum_{i=1}^n \lambda_i p_i :
\lambda_i \ge 0,\ 
\sum_{i=1}^n \lambda_i = 1
\right\}
\tag{21.4.3}
$$

This definition is algebraic, but it has direct geometric meaning. In two dimensions, $\operatorname{conv}(P)$ is a convex polygon whose vertices are a subset of $P$. In three dimensions, it is a convex polyhedron whose faces are planar polygons, often triangulated into facets for algorithmic convenience.

A point $p_i\in P$ is an extreme point of $\operatorname{conv}(P)$ if it cannot be written as a nontrivial convex combination of other points in the hull. Thus, $p_i$ is extreme if there do not exist coefficients $\lambda_j\ge 0$, with $\sum_{j\ne i}\lambda_j=1$, such that:

$$p_i=\sum_{j\ne i}\lambda_j p_j \tag{21.4.4}$$

Only extreme points appear as hull vertices. Interior points and points lying strictly inside hull edges or faces are geometrically redundant for the boundary representation, although they may still be important for the scientific data set itself.

A supporting hyperplane of a convex set $C\subset\mathbb{R}^d$ is a hyperplane:

$$H=\{x\in\mathbb{R}^d:n\cdot x=\alpha\}\tag{21.4.5}$$

such that,

$$n\cdot x\le \alpha\qquad\text{for all }x\in C \tag{21.4.6}$$

and

$$H\cap C\ne\varnothing \tag{21.4.7}$$

The vector $n$ is an outward normal for the supporting half-space. For a finite point set, a point $p_k\in P$ is exposed by a direction $n$ if:

$$n\cdot p_k = \max_{1\le i\le n} n\cdot p_i \tag{21.4.8}$$

This maximization view is useful in collision detection, bounding computations, support-function methods, and separating-axis arguments. The support function of a convex hull is:

$$h_P(n) = \max_{x\in \operatorname{conv}(P)} n\cdot x = \max_{1\le i\le n} n\cdot p_i \tag{21.4.9}$$

Thus, optimizing a linear functional over the hull reduces to checking the original points.

In scientific computing, the hull is often used not because the exact boundary of the data is sufficient, but because it provides a safe outer envelope. For a point cloud sampled from a physical surface or domain, the convex hull gives a first bounding approximation. For a set of particles, it provides a coarse occupied region. For a mesh, it can support fast rejection before more expensive point-in-cell or intersection tests are attempted. This is especially useful when hull computations are combined with spatial indexes or hierarchical bounding structures.

## 21.4.2. Two-Dimensional Convex Hull Algorithms

In two dimensions, the convex hull of $n$ points can be computed efficiently by sorting and orientation tests. The two most important textbook algorithms are Graham scan and the monotone chain algorithm. Both rely on the orientation predicate:

$$\operatorname{orient2d}(a,b,c) = \det\begin{bmatrix}a_x-c_x & a_y-c_y\\b_x-c_x & b_y-c_y\end{bmatrix} \tag{21.4.10}$$

or an equivalent determinant convention. The sign of this predicate determines whether a turn is left, right, or collinear.

The monotone chain algorithm first sorts the points lexicographically, usually by $x$-coordinate and then by $y$-coordinate:

$$
p_i < p_j
\quad\Longleftrightarrow\quad
p_{i,x} < p_{j,x}
\ \text{or}\
\left(p_{i,x}=p_{j,x}\ \text{and}\ p_{i,y}<p_{j,y}\right)
\tag{21.4.11}
$$

It then constructs the lower and upper hulls by scanning through the sorted list. Suppose the current partial hull ends with vertices $u,v$, and the next candidate point is $w$. If the orientation,

$$\operatorname{orient2d}(u,v,w)\tag{21.4.12}$$

has the wrong sign for the desired hull chain, then $v$ cannot remain on the convex boundary and is removed. This pop operation is repeated until convexity is restored. The same procedure is applied to construct both the lower and upper chains, which are then joined.

The essential invariant is that each consecutive triple of retained vertices must make a consistent turn. If the hull is oriented counterclockwise, this can be written schematically as:

$$\operatorname{orient2d}(h_{k-2},h_{k-1},h_k) \ge 0\tag{21.4.13}$$

or with the opposite sign depending on the chosen determinant convention. A strict inequality excludes collinear points along hull edges, while a non-strict inequality retains them according to the implementation policy. This policy must be chosen deliberately. For many scientific applications, keeping only extreme endpoints of collinear boundary chains is preferable because it gives a minimal hull. In other applications, retaining all boundary samples is useful because those points may carry physical or mesh-related data.

Graham scan is similar in spirit but begins by selecting a pivot point, often the point with lowest $y$-coordinate and then lowest $x$-coordinate. The remaining points are sorted by polar angle around the pivot. The algorithm then scans this angular ordering and removes points that violate the convex-turn condition. If $p_0$ is the pivot, the angular sorting can be expressed through comparisons of directions:

$$p_i-p_0,\qquad p_j-p_0 \tag{21.4.14}$$

using orientation predicates rather than explicit trigonometric angles. Avoiding angle computation is numerically and computationally preferable because it reduces dependence on inverse trigonometric functions and preserves the determinant-based structure of the algorithm.

Both Graham scan and monotone chain have sorting-dominated complexity,

$$O(n\log n)\tag{21.4.15}$$

After sorting, the scan itself is linear because each point is pushed and popped at most a constant number of times. Gift wrapping, also known as Jarvis march, provides an output-sensitive alternative. If $h$ is the number of hull vertices, gift wrapping has complexity,

$$O(nh) \tag{21.4.16}$$

It is attractive when $h\ll n$, but it can degrade to $O(n^2)$ when most points lie on the hull.

From a robustness viewpoint, all these algorithms depend on a consistent orientation predicate. Duplicate points should be removed or normalized before hull construction. Collinear points must be handled according to a documented policy. Nearly collinear points should not be classified by scattered ad hoc tolerances, because inconsistent turn decisions may produce self-intersecting or incomplete hull chains. Recent studies of convex-hull implementations continue to emphasize these issues, especially when comparing two-dimensional and three-dimensional behavior or when supporting dynamic updates (Gæde et al., 2024; Kwon, Oh and Baek, 2024).

### Rust Implementation

Following the discussion in Section 21.4.2 on determinant-based orientation predicates and sorting-based convex hull construction, Program 21.4.1 provides a practical implementation of the two-dimensional convex hull problem using the monotone chain algorithm. In computational geometry, convex hull construction forms a foundational preprocessing operation for collision detection, bounding-volume generation, point-cloud analysis, and geometric filtering. The implementation demonstrates how lexicographic sorting and repeated orientation testing can be combined to recover the outer convex envelope of a planar point set while excluding interior and redundant collinear points. The program also evaluates geometric diagnostics such as polygon orientation, signed area, and support-function maximization, thereby connecting the computational procedure directly to the mathematical concepts introduced in Equations (21.4.1)–(21.4.16). Particular emphasis is placed on robustness through duplicate removal, consistent orientation testing, and explicit handling of convex-turn conditions, all of which are essential for reliable scientific computing implementations.

At the core of the implementation is the `Point` structure, which represents planar coordinates and provides the basic geometric operations required throughout the convex hull computation. In addition to storing Cartesian coordinates `(x,y)`, the structure implements a dot-product operation used in evaluating the support function introduced in Equation (21.4.9). This abstraction allows the same geometric representation to be reused consistently across orientation tests, support-function evaluations, and polygon-area computations.

The `lexicographic_cmp` function implements the ordering relation described in Equation (21.4.11). The points are first sorted according to increasing x-coordinate and then by increasing y-coordinate when ties occur. This ordering is fundamental to the monotone chain algorithm because it guarantees that the lower and upper hull chains can be constructed through a single forward and reverse scan of the sorted point set. Before hull construction begins, duplicate points are removed to prevent degeneracies and inconsistent turn classifications during the scan process.

The geometric orientation test is implemented by the `orient2d` function, which evaluates the determinant expression introduced in Equation (21.4.10). The sign of this determinant determines whether three consecutive points form a left turn, right turn, or collinear configuration. Because convex hull construction depends entirely on maintaining consistent turning behavior along the boundary, the orientation predicate serves as the central geometric primitive of the entire algorithm. The determinant-based formulation avoids explicit angle computations and therefore preserves both computational efficiency and numerical robustness.

The principal hull-construction logic is implemented in the `monotone_chain_convex_hull` function. Following the mathematical discussion associated with Equations (21.4.12)–(21.4.13), the algorithm incrementally constructs lower and upper hull chains while repeatedly removing points that violate the convex-turn condition. Each new candidate point is tested against the two most recent hull vertices using the orientation predicate. If the resulting turn has the wrong sign, the intermediate vertex is removed through a pop operation until convexity is restored. This iterative removal process ensures that every consecutive triple of retained vertices maintains the required convex orientation. The implementation also provides a policy option for retaining or discarding collinear boundary points, reflecting the discussion of strict and non-strict inequalities in Equation (21.4.13).

The `signed_polygon_area` function computes the oriented area of the resulting hull polygon using the classical shoelace summation formula. The sign of the computed area determines whether the hull vertices are ordered clockwise or counterclockwise, while the magnitude provides the geometric area enclosed by the hull. This diagnostic is useful both for validating the correctness of the constructed hull and for detecting degenerate configurations in which the hull collapses to a line segment or point set.

The `support_point` function implements the support-function maximization process described in Equations (21.4.8)–(21.4.9). For a given direction vector `n`, the function evaluates the dot product `n·p_i` over all hull vertices and returns the point achieving the maximum value. This operation is central in collision detection, separating-axis methods, bounding-volume construction, and optimization-based geometric algorithms because it reduces linear optimization over the convex hull to a finite maximization over discrete points.

The `print_points` utility function provides formatted visualization of the input point set and the resulting convex hull vertices. Although simple, such diagnostic output is extremely useful in computational geometry because hull algorithms are highly sensitive to degeneracies, duplicate points, and orientation inconsistencies. Structured output therefore assists in validating algorithmic correctness during numerical experimentation and debugging.

The `main` function serves to demonstrate the complete convex hull pipeline using a representative planar point set containing interior points, duplicate samples, and collinear boundary points. It begins by defining a collection of test points and then invokes the `monotone_chain_convex_hull` function to compute the convex boundary in counterclockwise order. The resulting hull vertices are displayed together with geometric diagnostics including signed area, absolute area, and orientation classification. The program then evaluates the support function in several directions to identify exposed hull vertices associated with different outward normals. These computations illustrate the practical geometric interpretation of the support-function framework introduced in Section 21.4.1 and verify the correctness of the convex hull construction. Altogether, the implementation demonstrates how determinant-based predicates, sorting strategies, and geometric invariants combine to produce an efficient and robust convex hull algorithm suitable for scientific computing applications.

```rust
// Program 21.4.1: Two-Dimensional Convex Hull by the Monotone Chain Algorithm
//
// Problem statement:
// Given a finite set of planar points P = {p_1, p_2, ..., p_n}, compute the
// vertices of conv(P), the smallest convex polygon containing all points.
// The implementation follows the monotone chain algorithm: sort points
// lexicographically, remove duplicates, construct lower and upper hull chains
// using the orientation predicate, and join the chains into a counterclockwise
// convex hull. The program also evaluates the hull area and the support
// function h_P(n) = max_i n · p_i for selected directions.

use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn dot(self, other: Point) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

fn lexicographic_cmp(a: &Point, b: &Point) -> Ordering {
    match a.x.total_cmp(&b.x) {
        Ordering::Equal => a.y.total_cmp(&b.y),
        order => order,
    }
}

fn orient2d(a: Point, b: Point, c: Point) -> f64 {
    (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
}

fn monotone_chain_convex_hull(points: &[Point], keep_collinear: bool) -> Vec<Point> {
    if points.len() <= 1 {
        return points.to_vec();
    }

    let mut pts = points.to_vec();
    pts.sort_by(lexicographic_cmp);
    pts.dedup();

    if pts.len() <= 1 {
        return pts;
    }

    let should_pop = |a: Point, b: Point, c: Point| -> bool {
        let turn = orient2d(a, b, c);
        if keep_collinear {
            turn < 0.0
        } else {
            turn <= 0.0
        }
    };

    let mut lower: Vec<Point> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 {
            let n = lower.len();
            if should_pop(lower[n - 2], lower[n - 1], p) {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(p);
    }

    let mut upper: Vec<Point> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 {
            let n = upper.len();
            if should_pop(upper[n - 2], upper[n - 1], p) {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();

    lower.extend(upper);
    lower
}

fn signed_polygon_area(poly: &[Point]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..poly.len() {
        let p = poly[i];
        let q = poly[(i + 1) % poly.len()];
        sum += p.x * q.y - p.y * q.x;
    }

    0.5 * sum
}

fn support_point(points: &[Point], direction: Point) -> Option<(Point, f64)> {
    points
        .iter()
        .copied()
        .map(|p| (p, p.dot(direction)))
        .max_by(|(_, va), (_, vb)| va.total_cmp(vb))
}

fn print_points(title: &str, points: &[Point]) {
    println!("{title}");
    println!("{}", "-".repeat(title.len()));

    for (i, p) in points.iter().enumerate() {
        println!("p_{:<2} = ({:>10.6}, {:>10.6})", i, p.x, p.y);
    }

    println!("number of points = {}\n", points.len());
}

fn main() {
    let points = vec![
        Point::new(0.0, 0.0),
        Point::new(4.0, 0.0),
        Point::new(4.0, 3.0),
        Point::new(0.0, 3.0),
        Point::new(2.0, 1.5),
        Point::new(1.0, 1.0),
        Point::new(3.0, 2.0),
        Point::new(2.0, 0.0),
        Point::new(4.0, 1.5),
        Point::new(0.0, 0.0),
    ];

    println!("Two-Dimensional Convex Hull by Monotone Chain");
    println!("=============================================\n");

    print_points("Input Point Set", &points);

    let hull = monotone_chain_convex_hull(&points, false);

    print_points("Convex Hull Vertices", &hull);

    let signed_area = signed_polygon_area(&hull);
    println!("Hull Diagnostics");
    println!("----------------");
    println!("signed area = {:>.10}", signed_area);
    println!("area        = {:>.10}", signed_area.abs());
    println!(
        "orientation = {}\n",
        if signed_area > 0.0 {
            "counterclockwise"
        } else if signed_area < 0.0 {
            "clockwise"
        } else {
            "degenerate"
        }
    );

    let directions = vec![
        Point::new(1.0, 0.0),
        Point::new(0.0, 1.0),
        Point::new(1.0, 1.0),
        Point::new(-1.0, 1.0),
    ];

    println!("Support Function Values");
    println!("-----------------------");

    for dir in directions {
        if let Some((p, value)) = support_point(&hull, dir) {
            println!(
                "n = ({:>6.2}, {:>6.2})  h_P(n) = {:>10.6}  exposed point = ({:>8.4}, {:>8.4})",
                dir.x, dir.y, value, p.x, p.y
            );
        }
    }
}
```

Program 21.4.1 demonstrates a practical implementation of two-dimensional convex hull construction using determinant-based orientation predicates and lexicographic sorting. The monotone chain approach illustrates how geometric convexity constraints can be enforced incrementally through local turn tests while maintaining overall algorithmic efficiency. Because each point is inserted and removed at most a constant number of times after sorting, the implementation achieves the expected $O(n\log n)$ complexity discussed in Equation (21.4.15).

The numerical examples highlight several important geometric behaviors. Interior points are correctly excluded from the hull boundary, duplicate points are removed before processing, and collinear boundary samples are handled consistently according to the chosen convexity policy. The resulting hull therefore represents the minimal convex polygon enclosing the input point set. The support-function computations further demonstrate how convex hulls can be queried efficiently for directional extrema, an operation that plays a major role in optimization, collision detection, and spatial acceleration methods.

The modular structure of the implementation allows the framework to be extended naturally to more advanced geometric algorithms. Possible extensions include dynamic convex hull updates, higher-dimensional hull construction, rotating-calipers methods for diameter and width computations, and integration with hierarchical spatial data structures such as bounding-volume hierarchies or k-d trees. More advanced implementations may also incorporate adaptive exact predicates and robust arithmetic techniques to address the numerical challenges associated with nearly collinear or degenerate point configurations encountered in large-scale scientific computing applications.

## 21.4.3. Three-Dimensional Hulls and Half-Space Structure

In three dimensions, the convex hull of a finite point set is a convex polyhedron. Its boundary consists of vertices, edges, and faces. For algorithmic purposes, the faces are often triangulated into oriented facets. A triangular facet $(a,b,c)$ is part of the hull boundary if all other points lie on one side of its supporting plane:

$$\operatorname{orient3d}(a,b,c,p_i)\ge 0\quad\text{for all }p_i\in P \tag{21.4.17}$$

or with the opposite sign depending on the outward orientation convention. The sign consistency of this predicate determines whether the face is visible from a point and whether it belongs to the boundary of the hull.

A convex polyhedron can also be described as an intersection of half-spaces:

$$\operatorname{conv}(P) = \bigcap_{j=1}^m\{x\in\mathbb{R}^3:n_j\cdot x\le \alpha_j\},\tag{21.4.18}$$

where each inequality corresponds to a supporting plane of a hull face. This half-space representation is dual to the vertex representation in equation (21.4.3). The vertex representation is natural when the input is a point set; the half-space representation is natural for clipping, feasibility tests, contact, and separation.

Incremental three-dimensional hull algorithms insert points one at a time. Given an existing hull and a new point $p$, the algorithm finds all faces visible from $p$. A face $(a,b,c)$ is visible if $p$ lies outside the hull relative to that face:

$$\operatorname{orient3d}(a,b,c,p)\tag{21.4.19}$$

has the exterior sign. The boundary between visible and nonvisible faces forms a horizon. The visible faces are removed, and new faces are created by connecting $p$ to the horizon edges. This is conceptually simple, but robust implementation is substantially harder than in two dimensions because the algorithm must maintain a consistent facet-edge adjacency structure.

Degeneracies are also more complicated in three dimensions. Four coplanar points satisfy:

$$\operatorname{orient3d}(a,b,c,d)=0,\tag{21.4.20}$$

and many input points may lie on the same supporting plane. If the algorithm assumes triangular facets in general position, coplanar face sets must either be triangulated consistently or represented as polygonal faces. Otherwise, the hull may contain duplicate facets, missing faces, or inconsistent adjacency. Near-coplanar cases are even more delicate under floating-point arithmetic because the sign in (21.4.19) may not be reliable without a robust predicate.

The difficulty of three-dimensional hull construction is therefore not only asymptotic. It is topological. The algorithm must update a cell complex consisting of vertices, edges, and faces while maintaining orientation and adjacency. This makes three-dimensional hulls representative of a recurring theme in computational geometry for scientific computing: higher-dimensional algorithms often require the same determinant predicates as their two-dimensional counterparts, but the surrounding data structure is more fragile.

Although a full production-quality three-dimensional hull algorithm is beyond the role of an introductory section, the main mathematical ideas should be clear. Hull facets are supported by oriented planes. Visibility is determined by signed volumes. The hull is equivalently a convex combination set or an intersection of half-spaces. Robustness depends on consistent orientation decisions and careful handling of coplanar degeneracies. These observations connect directly to later topics such as Delaunay triangulations, Voronoi diagrams, polytope clipping, and mesh generation.

## 21.4.4. Hulls, Bounding, and Scientific-Computing Applications

Convex hulls are useful in scientific computing because they compress geometric information. The hull of a point set discards interior points and retains only the outer envelope:

$$P \subset \operatorname{conv}(P) \tag{21.4.21}$$

This makes the hull a conservative bounding region. If a query point $x$ lies outside $\operatorname{conv}(P)$, then it cannot lie inside any geometric object whose vertices are contained in $P$. Conversely, if $x$ lies inside the hull, further tests are still required because the original domain may be nonconvex. Thus, the hull is especially effective as a fast rejection test.

In point-cloud analysis, the convex hull provides a coarse shape descriptor. For a sampled set $P$, one may compute the hull area in two dimensions or hull volume in three dimensions as a rough measure of spatial spread. In three dimensions, if the hull boundary is triangulated into oriented facets $(a_j,b_j,c_j)$, the enclosed volume can be computed by summing signed tetrahedral volumes relative to a fixed origin:

$$
V
=
\frac{1}{6}
\sum_j
\det
\begin{bmatrix}
a_{j,x} & a_{j,y} & a_{j,z} \\
b_{j,x} & b_{j,y} & b_{j,z} \\
c_{j,x} & c_{j,y} & c_{j,z}
\end{bmatrix}
\tag{21.4.22}
$$

The absolute value is taken if the orientation of all facets is consistent but opposite to the chosen convention:

$$
|V|
=
\left|
\frac{1}{6}
\sum_j \det(a_j,b_j,c_j)
\right|
\tag{21.4.23}
$$

This formula is a direct extension of signed-volume ideas from Section 21.2. It is frequently useful in geometry processing and mesh validation, but it depends on the hull being closed and consistently oriented.

Convex hulls are also related to separation and collision detection. Two convex sets $A,B\subset\mathbb{R}^d$ are disjoint if there exists a hyperplane separating them. In support-function form, a direction $n$ separates them if:

$$\max_{x\in A} n\cdot x<\min_{y\in B} n\cdot y \tag{21.4.24}$$

For finite point sets, this becomes a comparison of support values:

$$h_A(n)<-h_B(-n)\tag{21.4.25}$$

where,

$$h_A(n)=\max_{x\in A}n\cdot x,\qquad h_B(-n)=\max_{y\in B}(-n)\cdot y \tag{21.4.26}$$

This support-function viewpoint links convex hulls to bounding volumes, contact algorithms, and optimization over convex polytopes.

In meshing, convex hulls often appear indirectly. Delaunay triangulations of point sets fill the convex hull of the input points unless constrained by domain boundaries. Therefore, if the physical domain is nonconvex, additional boundary constraints or clipping are required. In scattered-data interpolation, a query point outside the convex hull of the samples is an extrapolation point rather than an interpolation point. This distinction is crucial because interpolation weights based on barycentric coordinates may become negative outside the enclosing simplex or hull, changing the stability and interpretation of the approximation.

For finite element and finite volume workflows, convex hulls should therefore be understood as auxiliary geometric structures rather than complete domain descriptions. They are excellent for bounding, rejection, support queries, and shape summarization, but they cannot represent holes, concavities, internal boundaries, or material interfaces. Those features require polygonal or polyhedral arrangements, constrained triangulations, constructive solid geometry, or mesh-based topology. Still, because hulls are simple, determinant-based, and closely connected to orientation predicates, they form a natural bridge between primitive geometry and the more advanced structures developed later in the chapter.

### Rust Implementation

Following the discussion in Sections 21.4.3–21.4.5 on supporting planes, half-space representations, signed-volume computations, and support-function geometry, Program 21.4.2 provides a practical implementation of a triangulated three-dimensional convex hull representation together with several geometric operations derived directly from the underlying hull structure. In computational geometry for scientific computing, three-dimensional hulls are substantially more delicate than their two-dimensional counterparts because the algorithm must maintain oriented facets, preserve topological consistency, and handle coplanar degeneracies carefully. The implementation therefore emphasizes a consistently oriented triangular-facet representation of a cube-like hull and demonstrates how such a representation can support volume evaluation, half-space extraction, support-function queries, and separation testing. The program also illustrates the relationship between the vertex representation of a convex polyhedron and its dual half-space representation introduced in Equations (21.4.18)–(21.4.26). Particular emphasis is placed on orientation consistency and explicit facet construction because reliable geometric predicates and topological coherence are essential for robust scientific-computing workflows.

At the core of the implementation is the `Point3` structure, which represents points and vectors in three-dimensional Euclidean space. In addition to storing Cartesian coordinates `(x,y,z)`, the structure implements vector subtraction, dot products, cross products, scaling, and Euclidean norms. These operations form the basic algebraic framework required for orientation predicates, supporting-plane construction, support-function evaluation, and tetrahedral signed-volume computations. By encapsulating these operations inside a dedicated geometric type, the implementation preserves clarity while avoiding repeated low-level coordinate manipulations throughout the hull-processing pipeline.

The `Facet` structure represents an oriented triangular hull face through three vertex indices `(a,b,c)`. This orientation is essential because the sign conventions introduced in Equations (21.4.17)–(21.4.20) depend directly on consistent outward normals. The `HalfSpace` structure stores the normal vector and offset parameter associated with a supporting plane of the hull. Together, these structures provide a direct computational realization of the half-space representation introduced in Equation (21.4.18).

The geometric primitives implemented inside the `Point3` methods correspond directly to the determinant-based framework developed in the section text. The `sub` method constructs edge vectors between vertices, the `dot` method evaluates scalar products required in support-function computations, and the `cross` method computes oriented normal vectors associated with hull facets. The `norm` method provides Euclidean normalization for supporting-plane normals, while the `scale` method allows normalized outward directions to be constructed consistently. These vector operations collectively support all subsequent geometric calculations performed by the program.

The `tetra_signed_volume` function implements the signed tetrahedral-volume formula underlying Equation (21.4.22). For an oriented triangle `(a,b,c)` relative to the origin, the scalar triple product computes the signed volume contribution associated with the tetrahedron formed by the facet and the coordinate origin. Summing these contributions over all consistently oriented hull facets yields the total enclosed hull volume. Because the sign depends on orientation consistency, incorrect facet ordering would produce cancellation or erroneous volume estimates, illustrating the importance of coherent facet orientation in three-dimensional hull algorithms.

The `facet_half_space` function constructs the supporting half-space associated with an oriented triangular facet. Given three facet vertices, the function computes the outward normal using a cross product and normalizes the result before evaluating the supporting-plane offset parameter `alpha`. This directly realizes the half-space formulation introduced in Equation (21.4.18), where each hull face corresponds to a supporting plane of the convex polyhedron. The resulting half-space representation is useful for geometric clipping, collision testing, feasibility analysis, and point-classification operations.

The `cube_hull_facets` function provides an explicit triangulated representation of the cube boundary using twelve consistently oriented triangular facets. Each square face of the cube is decomposed into two triangles, thereby avoiding the coplanar overcounting issue discussed in Section 21.4.3. This explicit triangulation ensures that the hull forms a closed oriented surface suitable for signed-volume integration. The function therefore serves both as a geometric construction mechanism and as a demonstration of how coplanar polygonal faces must be triangulated consistently in practical hull implementations.

The `hull_volume` function evaluates the total convex-hull volume by summing tetrahedral signed-volume contributions over all oriented facets. This computation implements the signed-volume accumulation framework described in Equations (21.4.22)–(21.4.23). Because the hull facets are consistently oriented outward, the resulting total volume is positive and equal to the physical enclosed volume of the polyhedron. This function demonstrates the close relationship between oriented geometry and volumetric integration in computational geometry workflows.

The `support_point` function implements the support-function maximization process described in Equations (21.4.24)–(21.4.26). For a specified direction vector `n`, the function evaluates the dot product `n·p_i` over all hull vertices and returns the point maximizing this quantity. This operation is central in collision detection, separating-axis methods, convex optimization, and support-mapping algorithms because it reduces directional geometric queries to a finite maximization problem over hull vertices.

The `separated_by_direction` function demonstrates a simple support-function separation test between two convex point sets. By comparing support values along a candidate separating direction, the function determines whether a hyperplane orthogonal to that direction separates the two convex hulls. This provides a direct computational realization of the support-function inequalities introduced in Equations (21.4.24)–(21.4.26) and illustrates how convex geometry can be used for efficient collision rejection and bounding-volume testing.

The `print_points` utility function provides formatted diagnostic output for geometric debugging and verification. Although straightforward, such diagnostic visualization is extremely useful in computational geometry because orientation inconsistencies, duplicate facets, and degeneracies can otherwise be difficult to detect from numerical computations alone.

The `main` function serves to demonstrate the complete geometric workflow associated with three-dimensional convex hull processing. It begins by constructing a cube-like point set together with an interior sample point that should not appear on the hull boundary. The program then generates the triangulated hull facets, evaluates the enclosed hull volume, extracts supporting half-spaces for each facet, and computes support-function extrema in several directions. Finally, the implementation demonstrates a support-function-based separation test by comparing the hull against a translated copy shifted along the x-direction. The resulting output verifies the correctness of the facet orientations, supporting-plane normals, signed-volume accumulation, and support-function computations. Altogether, the program illustrates how oriented facets, determinant-based geometry, and convex support structures interact in practical scientific-computing applications involving three-dimensional geometric data.

```rust
// Program 21.4.2: Corrected Three-Dimensional Convex Hull Facets, Half-Spaces, and Support Queries
//
// Problem statement:
// Construct a consistently oriented triangular-facet representation of the
// convex hull of a cube-like three-dimensional point set. The program evaluates
// supporting half-spaces, signed volume, support points, and a simple
// support-function separation test.

#[derive(Clone, Copy, Debug, PartialEq)]
struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy, Debug)]
struct Facet {
    a: usize,
    b: usize,
    c: usize,
}

#[derive(Clone, Copy, Debug)]
struct HalfSpace {
    normal: Point3,
    alpha: f64,
}

impl Point3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn sub(self, other: Point3) -> Point3 {
        Point3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    fn dot(self, other: Point3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Point3) -> Point3 {
        Point3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    fn scale(self, s: f64) -> Point3 {
        Point3::new(s * self.x, s * self.y, s * self.z)
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
}

fn tetra_signed_volume(a: Point3, b: Point3, c: Point3) -> f64 {
    a.dot(b.cross(c)) / 6.0
}

fn facet_half_space(points: &[Point3], facet: Facet) -> HalfSpace {
    let a = points[facet.a];
    let b = points[facet.b];
    let c = points[facet.c];

    let raw_normal = b.sub(a).cross(c.sub(a));
    let length = raw_normal.norm();

    let normal = if length > 0.0 {
        raw_normal.scale(1.0 / length)
    } else {
        raw_normal
    };

    let alpha = normal.dot(a);

    HalfSpace { normal, alpha }
}

fn cube_hull_facets() -> Vec<Facet> {
    vec![
        // z = 0, outward normal (0, 0, -1)
        Facet { a: 0, b: 2, c: 1 },
        Facet { a: 0, b: 3, c: 2 },

        // z = 1, outward normal (0, 0, 1)
        Facet { a: 4, b: 5, c: 6 },
        Facet { a: 4, b: 6, c: 7 },

        // y = 0, outward normal (0, -1, 0)
        Facet { a: 0, b: 1, c: 5 },
        Facet { a: 0, b: 5, c: 4 },

        // y = 1, outward normal (0, 1, 0)
        Facet { a: 3, b: 7, c: 6 },
        Facet { a: 3, b: 6, c: 2 },

        // x = 0, outward normal (-1, 0, 0)
        Facet { a: 0, b: 4, c: 7 },
        Facet { a: 0, b: 7, c: 3 },

        // x = 1, outward normal (1, 0, 0)
        Facet { a: 1, b: 2, c: 6 },
        Facet { a: 1, b: 6, c: 5 },
    ]
}

fn hull_volume(points: &[Point3], facets: &[Facet]) -> f64 {
    let mut volume = 0.0;

    for f in facets {
        let a = points[f.a];
        let b = points[f.b];
        let c = points[f.c];

        volume += tetra_signed_volume(a, b, c);
    }

    volume.abs()
}

fn support_point(points: &[Point3], direction: Point3) -> Option<(usize, Point3, f64)> {
    points
        .iter()
        .copied()
        .enumerate()
        .map(|(i, p)| (i, p, p.dot(direction)))
        .max_by(|(_, _, va), (_, _, vb)| va.total_cmp(vb))
}

fn separated_by_direction(a: &[Point3], b: &[Point3], n: Point3) -> bool {
    let max_a = support_point(a, n).unwrap().2;
    let max_minus_b = support_point(b, n.scale(-1.0)).unwrap().2;

    max_a < -max_minus_b
}

fn print_points(title: &str, points: &[Point3]) {
    println!("{title}");
    println!("{}", "-".repeat(title.len()));

    for (i, p) in points.iter().enumerate() {
        println!(
            "p_{:<2} = ({:>8.4}, {:>8.4}, {:>8.4})",
            i, p.x, p.y, p.z
        );
    }

    println!("number of points = {}\n", points.len());
}

fn main() {
    let points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
        Point3::new(0.0, 1.0, 1.0),
        Point3::new(0.5, 0.5, 0.5),
    ];

    println!("Corrected Three-Dimensional Convex Hull Facets and Half-Spaces");
    println!("==============================================================\n");

    print_points("Input Point Set", &points);

    let facets = cube_hull_facets();

    println!("Triangular Hull Facets");
    println!("----------------------");
    for (i, f) in facets.iter().enumerate() {
        println!("facet_{:<2} = ({}, {}, {})", i, f.a, f.b, f.c);
    }
    println!("number of triangular facets = {}\n", facets.len());

    println!("Hull Volume");
    println!("-----------");
    println!("volume = {:>.10}\n", hull_volume(&points, &facets));

    println!("Supporting Half-Spaces");
    println!("----------------------");
    for (i, f) in facets.iter().enumerate() {
        let hs = facet_half_space(&points, *f);
        println!(
            "facet_{:<2}: n = ({:>8.4}, {:>8.4}, {:>8.4}), alpha = {:>8.4}",
            i, hs.normal.x, hs.normal.y, hs.normal.z, hs.alpha
        );
    }
    println!();

    println!("Support Function Values");
    println!("-----------------------");

    let directions = vec![
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(1.0, 1.0, 1.0),
    ];

    for d in directions {
        let (idx, p, value) = support_point(&points, d).unwrap();
        println!(
            "n = ({:>5.1}, {:>5.1}, {:>5.1})  h_P(n) = {:>8.4}  exposed p_{} = ({:>5.2}, {:>5.2}, {:>5.2})",
            d.x, d.y, d.z, value, idx, p.x, p.y, p.z
        );
    }

    println!();

    let shifted_points: Vec<Point3> = points
        .iter()
        .map(|p| Point3::new(p.x + 2.0, p.y, p.z))
        .collect();

    println!("Support-Function Separation Test");
    println!("--------------------------------");
    println!(
        "separated along n = (1,0,0): {}",
        separated_by_direction(&points, &shifted_points, Point3::new(1.0, 0.0, 0.0))
    );
}
```

Program 21.4.2 demonstrates a practical implementation of a triangulated three-dimensional convex hull representation together with supporting-plane extraction, signed-volume integration, and support-function geometry. The implementation reflects the central computational themes discussed in Sections 21.4.3–21.4.5: maintaining orientation consistency, constructing valid supporting half-spaces, and preserving topological coherence across hull facets.

The numerical example illustrates several important geometric behaviors. The interior point is correctly excluded from the hull boundary, the cube boundary is represented using a consistent triangulated surface of twelve facets, and the resulting signed-volume computation reproduces the correct physical volume of the cube. The support-function evaluations further demonstrate how directional extremal queries can be reduced to simple maximization problems over hull vertices, while the separation test illustrates the geometric meaning of supporting hyperplanes and convex-set separation.

The modular design of the implementation allows the framework to be extended naturally toward more advanced geometric algorithms. Possible extensions include fully incremental three-dimensional hull construction, robust coplanarity handling, adjacency reconstruction, horizon extraction, and integration with Delaunay triangulations or Voronoi structures. More sophisticated implementations may also incorporate adaptive exact predicates and topological mesh data structures to support large-scale scientific-computing workflows involving mesh generation, geometry processing, collision detection, and spatial search. Because convex hulls form one of the simplest nontrivial examples of topology-aware geometric computation, they provide a natural bridge between primitive determinant predicates and the more advanced spatial algorithms developed later in the chapter.

## 21.4.5. Robustness and Rust Implementation Notes

A reliable convex-hull implementation should begin by normalizing the input. Duplicate points should be removed. Nonfinite coordinates should be rejected or handled explicitly. A collinearity policy should be chosen before the scan begins. For example, in a two-dimensional hull, one may choose to keep only the endpoints of collinear boundary chains or to retain all boundary points. These two choices produce different but defensible outputs:

$$\text{minimal hull vertices}\qquad\text{or}\qquad\text{all boundary samples} \tag{21.4.27}$$

The choice should not be left to accidental floating-point behavior.

The orientation predicate should be inherited from the robust predicate layer rather than reimplemented locally. In two dimensions, every pop decision in Graham scan or monotone chain depends on:

$$\operatorname{sign}(\operatorname{orient2d}(u,v,w)) \tag{21.4.28}$$

In three dimensions, every visibility and coplanarity decision depends on,

$$\operatorname{sign}(\operatorname{orient3d}(a,b,c,p)) \tag{21.4.29}$$

If these signs are inconsistent, the hull algorithm may return a self-intersecting polygon, an open polyhedron, duplicate facets, or an incorrect boundary.

From a Rust design perspective, the hull code should be written around explicit types and classifications. A two-dimensional hull function should not expose orientation tolerances as scattered constants. Instead, it should accept or choose a predicate policy. A useful conceptual structure is:

$$\text{input points}\longrightarrow\text{normalization}\longrightarrow\text{sorted order}\longrightarrow\text{orientation-based scan}\\ \longrightarrow\text{documented collinearity policy}\tag{21.4.30}$$

For three-dimensional hulls, the corresponding structure is:

$$\text{points}\longrightarrow\text{oriented facets}\longrightarrow\text{visibility tests}\longrightarrow\text{horizon extraction}\\ \longrightarrow\text{adjacency reconstruction}.\tag{21.4.31}$$

The second pipeline is substantially more complex because it must maintain a topological data structure, not just an ordered list of boundary vertices.

For the purposes of this chapter, the most important lesson is that convex hulls are not isolated classical algorithms. They demonstrate how determinant predicates, sorting, degeneracy policy, and topology interact. They also prepare the reader for Delaunay triangulations, Voronoi diagrams, mesh generation, and spatial search. A hull is the simplest nontrivial example of a geometric algorithm whose mathematical definition is compact, whose computational realization depends on robust predicates, and whose scientific-computing value lies in the way it supports larger numerical workflows.

# 21.5. Delaunay Triangulation and Voronoi Diagrams

Delaunay triangulations and Voronoi diagrams form one of the central dual structures in computational geometry. They connect point sets, proximity, interpolation, mesh generation, and control-volume construction. In scientific computing, their importance comes not only from their geometric elegance but also from their practical role in scattered-data interpolation, unstructured meshing, nearest-neighbor graphs, remeshing, spatial discretization, and semi-discrete optimal transport. The Delaunay triangulation organizes points into simplices with favorable local geometric properties, while the Voronoi diagram partitions space into regions of nearest influence. Together, they provide a bridge between discrete samples and continuous geometric domains. Modern surveys and implementations emphasize that these structures remain active research objects, especially in large-scale, GPU-parallel, distributed, weighted, and robustness-sensitive settings (Elshakhs et al., 2024; Gao and Chen, 2025; Lévy et al., 2025).

## 21.5.1. Voronoi Cells and Delaunay Duality

Let,

$$P={p_1,p_2,\ldots,p_n}\subset \mathbb{R}^d\tag{21.5.1}$$

be a finite set of sites. The Voronoi cell associated with the site $p_i$ is the set of all points at least as close to $p_i$ as to every other site:

$$V_i = \{x\in\mathbb{R}^d:\|x-p_i\|_2\le \|x-p_j\|_2\ \text{for all }j=1,\ldots,n\}\tag{21.5.2}$$

The full Voronoi diagram is the collection:

$$\mathcal{V}(P)=\{V_i\}_{i=1}^n \tag{21.5.3}$$

Each Voronoi cell is an intersection of half-spaces. Indeed, the inequality,

$$\|x-p_i\|_2^2\le \|x-p_j\|_2^2\tag{21.5.4}$$

expands to,

$$x\cdot x-2p_i\cdot x+p_i\cdot p_i\leq x\cdot x-2p_j\cdot x+p_j\cdot p_j \tag{21.5.5}$$

and hence,

$$2(p_j-p_i)\cdot x\le\|p_j\|_2^2-\|p_i\|_2^2\tag{21.5.6}$$

Thus,

$$
V_i
=
\bigcap_{j\ne i}
\left\{
x\in\mathbb{R}^d :
2(p_j-p_i)\cdot x
\le
\|p_j\|_2^2-\|p_i\|_2^2
\right\}
\tag{21.5.7}
$$

This shows that Euclidean Voronoi cells are convex polyhedra, possibly unbounded.

The Delaunay triangulation is the geometric dual of the Voronoi diagram. In two dimensions, if two Voronoi cells $V_i$ and $V_j$ share an edge, then the sites $p_i$ and $p_j$ are connected by a Delaunay edge. If three Voronoi cells meet at a Voronoi vertex, then the corresponding three sites form a Delaunay triangle. More generally, in $\mathbb{R}^d$, a set of $d+1$ sites forms a Delaunay simplex when their Voronoi cells share a common point. Under nondegeneracy assumptions, this gives a simplicial complex whose vertices are the original sites.

The key geometric property is the empty-circumsphere condition. In two dimensions, a triangle $(p_i,p_j,p_k)$ is Delaunay if its circumcircle contains no other site from $P$ in its interior. In $d$ dimensions, a simplex,

$$\sigma=\operatorname{conv}\{p_{i_0},p_{i_1},\ldots,p_{i_d}\}\tag{21.5.8}$$

is Delaunay if there exists a closed ball $B(c,r)$ such that:

$$
\|p_{i_m}-c\|_2 = r,
\qquad
m=0,\ldots,d
\tag{21.5.9}
$$

and

$$
\|p_\ell-c\|_2 \ge r
\qquad
\text{for all } p_\ell \in P
\tag{21.5.10}
$$

If no point lies strictly inside the circumsphere, the simplex satisfies the Delaunay empty-sphere condition.

The Voronoi-Delaunay duality is especially useful in scientific computing because it connects cells of influence with simplicial interpolation. Voronoi cells define nearest-site regions and control volumes, while Delaunay simplices provide local interpolation domains. A point cloud can therefore be converted into a triangulation for interpolation, or into Voronoi cells for volume partitioning. This duality is one reason Delaunay and Voronoi structures appear repeatedly in scattered-data approximation, finite volume discretization, remeshing, and particle-to-mesh transfer.

### Rust Implementation

Following the discussion in Section 21.5.1 on Voronoi half-space intersections, nearest-site regions, and the duality between Voronoi diagrams and Delaunay triangulations, Program 21.5.1 provides a practical implementation of bounded Voronoi-cell construction in two dimensions using iterative half-space clipping. In computational geometry, Voronoi cells are among the most important proximity structures because they convert a discrete set of sites into a partition of continuous space according to nearest-neighbor influence. The implementation demonstrates how the nearest-site inequalities introduced in Equations (21.5.4)–(21.5.7) can be transformed directly into linear half-space constraints and intersected to produce convex polygonal cells. The program also verifies the geometric correctness of the resulting partition through nearest-site classification and area-conservation checks. Particular emphasis is placed on determinant-free geometric clipping, robust half-space evaluation, and explicit treatment of Voronoi vertices shared by multiple sites, all of which are important for reliable scientific-computing implementations.

At the core of the implementation is the `Point2` structure, which represents planar geometric coordinates and provides the vector operations required throughout the Voronoi construction process. In addition to storing Cartesian coordinates `(x,y)`, the structure implements dot products, squared Euclidean norms, and pairwise squared-distance computations. These operations are used repeatedly in nearest-site tests, Voronoi half-space generation, and geometric validation of the final cells. By encapsulating these operations inside a dedicated geometric type, the implementation avoids repeated coordinate-level calculations and preserves clarity throughout the clipping pipeline.

The `HalfSpace` structure stores the coefficients `(a,b,c)` of the linear inequality associated with a Voronoi bisector half-space. The `voronoi_half_space` function implements the algebraic construction derived from Equations (21.5.4)–(21.5.7). Given two sites `p_i` and `p_j`, the function constructs the linear inequality corresponding to the set of points at least as close to `p_i` as to `p_j`. This converts the nonlinear nearest-site condition into a half-space representation whose boundary is the perpendicular bisector between the two sites. Repeating this construction for all competing sites produces the complete Voronoi cell as an intersection of convex half-spaces.

The `satisfies_half_space` function evaluates whether a point satisfies a given Voronoi half-space inequality within a specified numerical tolerance. This function forms the basic geometric predicate used throughout the polygon-clipping process. Because floating-point computations near Voronoi boundaries may produce small numerical perturbations, the implementation incorporates a tolerance parameter `eps` to avoid unstable boundary classifications during clipping.

The `line_intersection` function computes the intersection point between a polygon edge and a clipping line. When an edge crosses a Voronoi bisector boundary, the function determines the interpolation parameter along the segment and constructs the corresponding intersection point. This operation is essential for maintaining valid polygonal Voronoi cells during successive clipping operations because new Voronoi vertices are created precisely at these boundary intersections.

The `clip_polygon_by_half_space` function implements the central geometric operation of the program: iterative polygon clipping against a convex half-space. Following the classical Sutherland–Hodgman clipping strategy, the function processes polygon edges one at a time and applies four possible inside-outside transition cases. If both endpoints lie inside the half-space, the edge endpoint is retained. If the edge crosses the clipping boundary, the corresponding intersection point is inserted. Through repeated clipping against all competing Voronoi half-spaces, the algorithm incrementally constructs the bounded Voronoi polygon associated with a single generating site.

The `bounded_voronoi_cell` function orchestrates the complete Voronoi-cell construction process. Beginning with an initial rectangular bounding box, the function clips the polygon successively against all half-spaces generated by competing sites. Each clipping operation reduces the feasible region until the remaining polygon corresponds to the bounded approximation of the Voronoi cell associated with the chosen site. This procedure provides a direct computational realization of the half-space intersection formulation introduced in Equation (21.5.7).

The `polygon_area` function evaluates the area of the resulting Voronoi polygon using the standard shoelace summation formula. Because Voronoi cells form a partition of the bounding region, the total area of all bounded cells should equal the area of the enclosing box. The area computation therefore serves both as a geometric diagnostic and as a verification of clipping consistency.

The `nearest_site_indices` function determines all sites that are nearest to a specified Voronoi vertex within a tolerance threshold. Unlike a simple nearest-neighbor query returning a single site, this implementation correctly identifies Voronoi boundary vertices that are equidistant from multiple generating sites. This is especially important because Voronoi vertices are typically intersections of multiple bisectors and therefore correspond to points shared by two or more Voronoi regions. The associated `format_site_indices` function converts these index sets into readable diagnostic output.

The `print_sites` and `print_cell` utility functions provide structured visualization of the input sites and the resulting Voronoi polygons. The `print_cell` function additionally reports the nearest-site sets associated with each Voronoi vertex together with polygonal area information. Such diagnostic output is extremely useful in computational geometry because clipping algorithms and geometric predicates are highly sensitive to degeneracies, coincident intersections, and floating-point boundary effects.

The `main` function serves to demonstrate the complete Voronoi-cell construction pipeline for a representative two-dimensional site configuration. It begins by defining a set of generating sites together with a rectangular bounding region that truncates the otherwise potentially unbounded Voronoi cells. The program then constructs each Voronoi cell through iterative half-space clipping, computes the corresponding polygonal area, and identifies the nearest-site sets associated with each Voronoi vertex. Finally, the implementation performs a partition-consistency check by summing all bounded-cell areas and comparing the result against the total area of the bounding box. The resulting output verifies the correctness of the half-space construction, polygon clipping, nearest-site classification, and area partitioning. Altogether, the implementation demonstrates how Voronoi geometry can be realized directly through convex half-space intersections without requiring explicit Delaunay triangulation or determinant-based circumcircle computations.

```rust
// Program 21.5.1: Voronoi Cell Construction as an Intersection of Half-Spaces
//
// Problem statement:
// Given a finite set of two-dimensional sites, construct a bounded
// approximation of each Voronoi cell by intersecting an initial rectangular
// box with the half-spaces generated by all competing sites. Each half-space
// follows directly from the expanded nearest-site inequality in Equation
// (21.5.7). The program prints each polygonal Voronoi cell and reports all
// sites that are equidistant, up to tolerance, from each Voronoi vertex.

#[derive(Clone, Copy, Debug, PartialEq)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct HalfSpace {
    a: f64,
    b: f64,
    c: f64,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn dot(self, other: Point2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    fn distance_squared(self, other: Point2) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

fn voronoi_half_space(site_i: Point2, site_j: Point2) -> HalfSpace {
    HalfSpace {
        a: 2.0 * (site_j.x - site_i.x),
        b: 2.0 * (site_j.y - site_i.y),
        c: site_j.norm_squared() - site_i.norm_squared(),
    }
}

fn satisfies_half_space(p: Point2, h: HalfSpace, eps: f64) -> bool {
    h.a * p.x + h.b * p.y <= h.c + eps
}

fn line_intersection(p: Point2, q: Point2, h: HalfSpace) -> Point2 {
    let fp = h.a * p.x + h.b * p.y - h.c;
    let fq = h.a * q.x + h.b * q.y - h.c;
    let t = fp / (fp - fq);

    Point2::new(p.x + t * (q.x - p.x), p.y + t * (q.y - p.y))
}

fn clip_polygon_by_half_space(polygon: &[Point2], h: HalfSpace, eps: f64) -> Vec<Point2> {
    if polygon.is_empty() {
        return Vec::new();
    }

    let mut clipped = Vec::new();

    for i in 0..polygon.len() {
        let current = polygon[i];
        let next = polygon[(i + 1) % polygon.len()];

        let current_inside = satisfies_half_space(current, h, eps);
        let next_inside = satisfies_half_space(next, h, eps);

        match (current_inside, next_inside) {
            (true, true) => {
                clipped.push(next);
            }
            (true, false) => {
                clipped.push(line_intersection(current, next, h));
            }
            (false, true) => {
                clipped.push(line_intersection(current, next, h));
                clipped.push(next);
            }
            (false, false) => {}
        }
    }

    clipped
}

fn bounded_voronoi_cell(
    sites: &[Point2],
    site_index: usize,
    bounding_box: &[Point2],
    eps: f64,
) -> Vec<Point2> {
    let mut cell = bounding_box.to_vec();
    let site_i = sites[site_index];

    for (j, &site_j) in sites.iter().enumerate() {
        if j == site_index {
            continue;
        }

        let h = voronoi_half_space(site_i, site_j);
        cell = clip_polygon_by_half_space(&cell, h, eps);

        if cell.is_empty() {
            break;
        }
    }

    cell
}

fn polygon_area(poly: &[Point2]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }

    let mut sum = 0.0;

    for i in 0..poly.len() {
        let p = poly[i];
        let q = poly[(i + 1) % poly.len()];
        sum += p.x * q.y - p.y * q.x;
    }

    0.5 * sum.abs()
}

fn nearest_site_indices(sites: &[Point2], x: Point2, eps: f64) -> Vec<usize> {
    let mut min_dist = f64::INFINITY;

    for &site in sites {
        min_dist = min_dist.min(x.distance_squared(site));
    }

    let mut nearest = Vec::new();

    for (i, &site) in sites.iter().enumerate() {
        let d2 = x.distance_squared(site);
        if (d2 - min_dist).abs() <= eps {
            nearest.push(i);
        }
    }

    nearest
}

fn format_site_indices(indices: &[usize]) -> String {
    indices
        .iter()
        .map(|i| format!("p_{}", i))
        .collect::<Vec<_>>()
        .join(", ")
}

fn print_sites(sites: &[Point2]) {
    println!("Sites");
    println!("-----");

    for (i, p) in sites.iter().enumerate() {
        println!("p_{:<2} = ({:>8.4}, {:>8.4})", i, p.x, p.y);
    }

    println!();
}

fn print_cell(site_index: usize, cell: &[Point2], sites: &[Point2], eps: f64) {
    println!("Voronoi Cell V_{}", site_index);
    println!("----------------");

    if cell.is_empty() {
        println!("cell is empty\n");
        return;
    }

    for (k, v) in cell.iter().enumerate() {
        let nearest = nearest_site_indices(sites, *v, eps);
        println!(
            "v_{:<2} = ({:>10.6}, {:>10.6})   nearest site set = {{{}}}",
            k,
            v.x,
            v.y,
            format_site_indices(&nearest)
        );
    }

    println!("number of vertices = {}", cell.len());
    println!("bounded cell area  = {:>.10}\n", polygon_area(cell));
}

fn main() {
    let sites = vec![
        Point2::new(0.25, 0.25),
        Point2::new(0.75, 0.30),
        Point2::new(0.55, 0.80),
        Point2::new(0.20, 0.70),
    ];

    let bounding_box = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.0, 1.0),
        Point2::new(0.0, 1.0),
    ];

    let eps = 1.0e-10;

    println!("Voronoi Cells as Intersections of Half-Spaces");
    println!("=============================================\n");

    print_sites(&sites);

    println!("Bounding Box");
    println!("------------");
    for (i, p) in bounding_box.iter().enumerate() {
        println!("b_{:<2} = ({:>8.4}, {:>8.4})", i, p.x, p.y);
    }
    println!();

    let mut total_area = 0.0;

    for i in 0..sites.len() {
        let cell = bounded_voronoi_cell(&sites, i, &bounding_box, eps);
        total_area += polygon_area(&cell);
        print_cell(i, &cell, &sites, eps);
    }

    println!("Partition Check");
    println!("---------------");
    println!("sum of bounded cell areas = {:>.10}", total_area);
    println!("bounding-box area         = {:>.10}", 1.0);
    println!(
        "absolute difference       = {:>.3e}",
        (total_area - 1.0_f64).abs()
    );
}
```

Program 21.5.1 demonstrates a practical implementation of Voronoi-cell construction through repeated half-space intersection and polygon clipping. The implementation reflects the central geometric idea developed in Section 21.5.1: a Voronoi region can be interpreted not merely as a proximity concept, but as a convex polyhedral intersection generated by nearest-site inequalities. This perspective is especially valuable in scientific computing because it connects nearest-neighbor geometry directly to convex clipping, control-volume construction, and spatial partitioning.

The numerical example illustrates several important geometric behaviors. Voronoi boundary vertices correctly identify multiple equidistant generating sites, the resulting polygonal cells partition the bounding region exactly within floating-point precision, and the clipping process produces convex polygonal cells whose boundaries coincide with nearest-site bisectors. These properties demonstrate the geometric consistency of the half-space formulation and highlight the dual relationship between Voronoi partitions and Delaunay connectivity structures.

The modular design of the implementation allows the framework to be extended naturally toward more advanced Voronoi and Delaunay algorithms. Possible extensions include unbounded-cell representations, weighted Voronoi diagrams, power diagrams, incremental Delaunay triangulation, Bowyer-Watson retriangulation, and GPU-parallel proximity structures. More sophisticated implementations may also incorporate robust exact predicates, adjacency structures, and hierarchical spatial search to support large-scale scattered-data interpolation, finite-volume discretization, mesh generation, and semi-discrete optimal transport. Because Voronoi geometry transforms discrete point samples into continuous spatial partitions, it forms one of the most important geometric bridges between point-cloud data and numerical discretization methods used throughout scientific computing.

## 21.5.2. Incircle Predicates, Edge Flips, and Local Delaunay Conditions

In two dimensions, the Delaunay condition can be tested locally by the incircle predicate introduced in Section 21.2. Suppose two adjacent triangles share the edge $[a,b]$:

$$T_1=(a,b,c),\qquad T_2=(b,a,d) \tag{21.5.11}$$

The edge $[a,b]$ is locally Delaunay if the point (d) does not lie inside the circumcircle of (a,b,c), equivalently if the corresponding incircle predicate has the non-interior sign under the chosen orientation convention:

$$\operatorname{incircle}(a,b,c,d)\ \text{has the Delaunay-admissible sign} \tag{21.5.12}$$

Using the translated determinant form,

$$
\operatorname{incircle}(a,b,c,d) = 
\det\begin{bmatrix}a_x-d_x & a_y-d_y & \|a-d\|_2^2\\ b_x-d_x & b_y-d_y & \|b-d\|_2^2\\c_x-d_x & c_y-d_y & \|c-d\|_2^2\end{bmatrix} \tag{21.5.13}
$$

If the edge is not locally Delaunay and the quadrilateral $(a,c,b,d)$ is convex, the edge can be flipped. The original diagonal $[a,b]$ is replaced by the alternative diagonal $[c,d]$, changing the two triangles from:

$$(a,b,c),\qquad (b,a,d)\tag{21.5.14}$$

to

$$(c,d,b),\qquad (d,c,a) \tag{21.5.15}$$

with orientations chosen consistently. The purpose of the flip is to restore the empty-circumcircle condition locally.

Edge flipping illustrates the dependence of triangulation algorithms on predicate signs. The topological operation itself is discrete: one diagonal is removed and another is inserted. The numerical decision that triggers it is a determinant sign. If the sign is wrong near cocircularity, the triangulation may become inconsistent, non-Delaunay, or dependent on accidental roundoff. Degenerate cases occur when,

$$\operatorname{incircle}(a,b,c,d)=0 \tag{21.5.16}$$

which means that the four points are cocircular. In this case, both diagonals may be Delaunay, and the algorithm must apply a deterministic tie-breaking rule if a unique triangulation is required.

The relationship between Delaunay triangulation and angle quality is also important. In two dimensions, the Delaunay triangulation maximizes the minimum angle among all triangulations of a fixed point set in a lexicographic sense, under standard nondegeneracy assumptions. This property helps reduce extremely skinny triangles, although it does not eliminate poor elements when the input point distribution itself is unfavorable. For a triangle with side lengths $a,b,c$ and area $A$, its circumradius is:

$$R=\frac{abc}{4A} \tag{21.5.17}$$

and its inradius is:

$$r=\frac{2A}{a+b+c} \tag{21.5.18}$$

A common dimensionless quality indicator is:

$$q=\frac{2r}{R} \tag{21.5.19}$$

For an equilateral triangle, $q=1$, while $q\to 0$ for increasingly degenerate triangles. Delaunay triangulations are therefore attractive for meshing and interpolation because their local empty-circle condition tends to improve angular quality relative to arbitrary triangulations.

In three dimensions, the analogous empty-circumsphere test uses the insphere predicate. However, Delaunay tetrahedralization is more complicated than planar triangulation because poor-quality sliver tetrahedra can satisfy the Delaunay condition. A sliver may have nearly coplanar vertices, small volume, and poor numerical quality despite having a relatively acceptable circumsphere configuration. Thus, in three-dimensional meshing, Delaunay tetrahedralization is often combined with refinement, optimization, sliver removal, or quality-control strategies. Modern Delaunay literature reflects this distinction between elegant geometric criteria and the more demanding requirements of scientific simulation (Elshakhs et al., 2024; Gao and Chen, 2025).

### Rust Implementation

Following the discussion in Section 21.5.2 on local Delaunay conditions, incircle predicates, and edge-flipping operations, Program 21.5.2 provides a practical implementation of determinant-based Delaunay edge testing and local triangulation repair in two dimensions. In computational geometry, Delaunay triangulations are constructed and maintained through repeated local geometric decisions based on orientation and incircle predicates. The implementation demonstrates how the determinant expression introduced in Equation (21.5.13) can be used to determine whether a shared edge violates the empty-circumcircle condition and should therefore be replaced by an alternative diagonal through an edge-flip operation. The program also evaluates geometric quality indicators associated with the original and flipped triangulations using the inradius-circumradius ratio introduced in Equation (21.5.19). Particular emphasis is placed on orientation consistency, convexity verification, and post-flip Delaunay validation because triangulation algorithms depend critically on reliable predicate evaluation near degenerate geometric configurations.

At the core of the implementation is the `Point2` structure, which represents planar geometric coordinates and provides the basic Euclidean operations required throughout the local Delaunay analysis. In addition to storing Cartesian coordinates `(x,y)`, the structure implements a distance computation used in evaluating triangle side lengths for circumradius and inradius calculations. This abstraction allows the same geometric representation to be reused consistently across orientation predicates, incircle evaluations, and mesh-quality computations.

The `Triangle` structure stores an oriented planar triangle through its three vertices `(a,b,c)`. Maintaining orientation consistency is essential because the sign of both the orientation predicate and the incircle determinant depends directly on vertex ordering. The implementation therefore uses orientation-aware utilities to ensure that all triangles remain consistently counterclockwise throughout the flipping process.

The `orient2d` function implements the determinant-based orientation predicate introduced earlier in Section 21.2. Given three points, the function evaluates the signed area determinant that determines whether the ordered triple forms a counterclockwise turn, clockwise turn, or degenerate collinear configuration. This predicate forms the foundation of virtually all planar Delaunay algorithms because consistent orientation is required before incircle signs can be interpreted correctly.

The `incircle_raw` function implements the translated determinant form of the incircle predicate described in Equation (21.5.13). By translating all coordinates relative to the query point `d`, the determinant evaluates whether `d` lies inside, outside, or exactly on the circumcircle of triangle `(a,b,c)`. Because the determinant involves only translated coordinates and squared Euclidean distances, the implementation directly reflects the algebraic structure of the empty-circumcircle condition developed in Section 21.5.2.

The `oriented_incircle` function adjusts the raw determinant sign according to the orientation of the triangle `(a,b,c)`. Since the sign convention for the incircle test depends on whether the reference triangle is oriented clockwise or counterclockwise, the function first evaluates the orientation predicate and then applies the appropriate sign correction. This produces a consistent orientation-independent interpretation of the local Delaunay condition introduced in Equation (21.5.12).

The `point_inside_circumcircle` and `locally_delaunay` functions implement the logical interpretation of the incircle determinant. The first determines whether the opposite point lies strictly inside the circumcircle of the reference triangle, while the second determines whether the shared edge satisfies the local Delaunay condition. These functions therefore translate determinant signs directly into discrete triangulation decisions. Because floating-point computations near cocircularity may be numerically unstable, the implementation incorporates a tolerance parameter `eps` to avoid unreliable sign classifications near the degenerate condition described in Equation (21.5.16).

The `quadrilateral_is_convex` function verifies whether the two adjacent triangles form a valid convex quadrilateral suitable for edge flipping. The function first checks that the opposite vertices lie on opposite sides of the shared edge `[a,b]`, ensuring that the edge is an interior diagonal rather than a boundary edge. It then evaluates orientation signs around the quadrilateral boundary order `a -> d -> b -> c` to confirm convexity. This verification is essential because an edge flip is geometrically meaningful only when the union of the two adjacent triangles forms a convex quadrilateral.

The `ensure_ccw` function enforces consistent counterclockwise orientation for all generated triangles. If a triangle is detected to have negative orientation, the function swaps two vertices to restore positive orientation. This step is crucial because the validity of subsequent incircle predicates and area computations depends directly on consistent orientation conventions.

The `flip_edge` function implements the discrete topological operation described in Equations (21.5.14)–(21.5.15). Given two adjacent triangles sharing the diagonal `[a,b]`, the function removes this diagonal and replaces it with the alternative diagonal `[c,d]`, thereby producing two new triangles. The resulting triangles are then reoriented consistently using the `ensure_ccw` function. This operation demonstrates the fundamental structure of local Delaunay optimization algorithms, where triangulations are incrementally improved through repeated local edge replacements.

The `triangle_area` function computes the geometric area of a triangle using the determinant-based orientation formula. The `triangle_quality` function then evaluates the quality indicator introduced in Equations (21.5.17)–(21.5.19). Using the triangle side lengths and area, the function computes the circumradius `R`, the inradius `r`, and finally the dimensionless quality ratio `q = 2r/R`. This quality measure approaches unity for equilateral triangles and decreases toward zero for increasingly degenerate configurations. The implementation therefore connects local Delaunay conditions directly to mesh-quality improvement.

The `print_point`, `print_triangle`, and `print_delaunay_test` utility functions provide structured diagnostic output for geometric debugging and numerical verification. These functions display orientation signs, triangle areas, quality indicators, and incircle evaluations in a readable form. Such diagnostic reporting is especially important in computational geometry because triangulation algorithms are highly sensitive to orientation inconsistencies, near-degenerate predicates, and floating-point sign errors.

The `main` function serves to demonstrate the complete local Delaunay repair process for a representative convex quadrilateral. It begins by defining four planar points such that the shared diagonal `[a,b]` violates the local empty-circumcircle condition. The program constructs the two initial adjacent triangles, evaluates the incircle determinant, verifies convexity, and determines whether the shared edge is locally Delaunay. Because the opposite point lies inside the circumcircle of the reference triangle, the implementation performs an edge flip replacing `[a,b]` by `[c,d]`. The resulting triangulation is then validated through post-flip incircle tests and triangle-quality evaluations. The output demonstrates that the flipped configuration satisfies the local Delaunay condition and generally improves the geometric quality of the triangulation. Altogether, the implementation illustrates how determinant predicates and discrete topological operations interact to maintain Delaunay structure in practical triangulation algorithms.

```rust
// Program 21.5.2: Incircle Predicate, Local Delaunay Test, and Edge Flip
//
// Problem statement:
// Given two adjacent triangles sharing an interior edge [a,b], test whether
// the shared edge is locally Delaunay by evaluating the incircle predicate.
// If the opposite point d lies inside the circumcircle of triangle (a,b,c),
// and the quadrilateral is convex, the shared diagonal [a,b] is replaced by
// the alternative diagonal [c,d]. The program also computes triangle-quality
// indicators before and after the flip using q = 2r/R.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    a: Point2,
    b: Point2,
    c: Point2,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance(self, other: Point2) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn incircle_raw(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let ax = a.x - d.x;
    let ay = a.y - d.y;
    let bx = b.x - d.x;
    let by = b.y - d.y;
    let cx = c.x - d.x;
    let cy = c.y - d.y;

    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;

    ax * (by * c2 - b2 * cy)
        - ay * (bx * c2 - b2 * cx)
        + a2 * (bx * cy - by * cx)
}

fn oriented_incircle(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let det = incircle_raw(a, b, c, d);

    if orient2d(a, b, c) > 0.0 {
        det
    } else {
        -det
    }
}

fn point_inside_circumcircle(a: Point2, b: Point2, c: Point2, d: Point2, eps: f64) -> bool {
    oriented_incircle(a, b, c, d) > eps
}

fn locally_delaunay(a: Point2, b: Point2, c: Point2, d: Point2, eps: f64) -> bool {
    !point_inside_circumcircle(a, b, c, d, eps)
}

fn quadrilateral_is_convex(a: Point2, b: Point2, c: Point2, d: Point2, eps: f64) -> bool {
    // The adjacent triangles are (a,b,c) and (b,a,d).
    // Therefore c and d must lie on opposite sides of the shared edge [a,b],
    // and the boundary order is a -> d -> b -> c.
    let side_c = orient2d(a, b, c);
    let side_d = orient2d(a, b, d);

    if side_c.abs() <= eps || side_d.abs() <= eps {
        return false;
    }

    if side_c * side_d >= 0.0 {
        return false;
    }

    let poly = [a, d, b, c];

    let mut positive = false;
    let mut negative = false;

    for i in 0..4 {
        let p = poly[i];
        let q = poly[(i + 1) % 4];
        let r = poly[(i + 2) % 4];

        let s = orient2d(p, q, r);

        if s > eps {
            positive = true;
        } else if s < -eps {
            negative = true;
        }
    }

    positive ^ negative
}

fn ensure_ccw(t: Triangle) -> Triangle {
    if orient2d(t.a, t.b, t.c) >= 0.0 {
        t
    } else {
        Triangle {
            a: t.a,
            b: t.c,
            c: t.b,
        }
    }
}

fn flip_edge(a: Point2, b: Point2, c: Point2, d: Point2) -> (Triangle, Triangle) {
    // Replace diagonal [a,b] by [c,d].
    let t1 = ensure_ccw(Triangle { a: c, b: d, c: b });
    let t2 = ensure_ccw(Triangle { a: d, b: c, c: a });

    (t1, t2)
}

fn triangle_area(t: Triangle) -> f64 {
    0.5 * orient2d(t.a, t.b, t.c).abs()
}

fn triangle_quality(t: Triangle) -> f64 {
    let side_a = t.b.distance(t.c);
    let side_b = t.a.distance(t.c);
    let side_c = t.a.distance(t.b);

    let area = triangle_area(t);

    if area <= 0.0 {
        return 0.0;
    }

    let circumradius = side_a * side_b * side_c / (4.0 * area);
    let inradius = 2.0 * area / (side_a + side_b + side_c);

    2.0 * inradius / circumradius
}

fn print_point(label: &str, p: Point2) {
    println!("{label} = ({:>8.4}, {:>8.4})", p.x, p.y);
}

fn print_triangle(label: &str, t: Triangle) {
    println!("{label}");
    println!("{}", "-".repeat(label.len()));
    print_point("a", t.a);
    print_point("b", t.b);
    print_point("c", t.c);
    println!("orientation = {:>.10}", orient2d(t.a, t.b, t.c));
    println!("area        = {:>.10}", triangle_area(t));
    println!("quality q   = {:>.10}\n", triangle_quality(t));
}

fn print_delaunay_test(label: &str, a: Point2, b: Point2, c: Point2, d: Point2, eps: f64) {
    let inc = oriented_incircle(a, b, c, d);
    let inside = point_inside_circumcircle(a, b, c, d, eps);
    let local = locally_delaunay(a, b, c, d, eps);

    println!("{label}");
    println!("{}", "-".repeat(label.len()));
    println!("oriented incircle = {:>.12e}", inc);
    println!("opposite point inside circumcircle = {}", inside);
    println!("locally Delaunay = {}\n", local);
}

fn main() {
    // These four points define a convex quadrilateral with [a,b] as the
    // interior diagonal. The point d lies inside the circumcircle of (a,b,c),
    // so [a,b] is not locally Delaunay and should be flipped to [c,d].
    let a = Point2::new(0.0, 0.0);
    let b = Point2::new(1.0, 1.0);
    let c = Point2::new(0.0, 1.0);
    let d = Point2::new(1.0, 0.2);

    let eps = 1.0e-12;

    let t1 = ensure_ccw(Triangle { a, b, c });
    let t2 = ensure_ccw(Triangle { a: b, b: a, c: d });

    println!("Incircle Predicate, Local Delaunay Test, and Edge Flip");
    println!("======================================================\n");

    println!("Input Points");
    println!("------------");
    print_point("a", a);
    print_point("b", b);
    print_point("c", c);
    print_point("d", d);
    println!();

    print_triangle("Initial Triangle T_1 = (a,b,c)", t1);
    print_triangle("Initial Triangle T_2 = (b,a,d)", t2);

    let inc = oriented_incircle(a, b, c, d);
    let inside = point_inside_circumcircle(a, b, c, d, eps);
    let is_delaunay = locally_delaunay(a, b, c, d, eps);
    let is_convex = quadrilateral_is_convex(a, b, c, d, eps);

    println!("Local Delaunay Diagnostics for Initial Edge [a,b]");
    println!("-------------------------------------------------");
    println!("oriented incircle(a,b,c,d) = {:>.12e}", inc);
    println!("d inside circumcircle(a,b,c) = {}", inside);
    println!("quadrilateral boundary order  = a -> d -> b -> c");
    println!("quadrilateral convex          = {}", is_convex);
    println!("edge [a,b] locally Delaunay   = {}\n", is_delaunay);

    if !is_delaunay && is_convex {
        let (new_t1, new_t2) = flip_edge(a, b, c, d);

        println!("Edge Flip Performed");
        println!("-------------------");
        println!("old diagonal = [a,b]");
        println!("new diagonal = [c,d]\n");

        print_triangle("Flipped Triangle T_1 = (c,d,b)", new_t1);
        print_triangle("Flipped Triangle T_2 = (d,c,a)", new_t2);

        print_delaunay_test(
            "Post-Flip Test: Opposite Point a Against Triangle (c,d,b)",
            c,
            d,
            b,
            a,
            eps,
        );

        print_delaunay_test(
            "Post-Flip Test: Opposite Point b Against Triangle (d,c,a)",
            d,
            c,
            a,
            b,
            eps,
        );
    } else if is_delaunay {
        println!("No edge flip is required because [a,b] is already locally Delaunay.");
    } else {
        println!("No edge flip is performed because the quadrilateral is not convex.");
    }
}
```

Program 21.5.2 demonstrates a practical implementation of local Delaunay testing and edge flipping using determinant-based geometric predicates. The implementation reflects one of the central ideas of computational geometry: sophisticated topological modifications can often be driven entirely by the signs of low-dimensional determinants. In Delaunay triangulation algorithms, the discrete operation of replacing one diagonal by another is governed entirely by the numerical evaluation of the incircle predicate.

The numerical example illustrates several important geometric behaviors. The initial diagonal `[a,b]` violates the local empty-circumcircle condition because the opposite point lies inside the circumcircle of the adjacent triangle. The quadrilateral convexity test confirms that an edge flip is geometrically admissible, and the replacement diagonal `[c,d]` restores the local Delaunay condition. The post-flip quality indicators further demonstrate how local Delaunay optimization tends to improve angular quality by reducing skinny triangle configurations.

The modular design of the implementation allows the framework to be extended naturally toward more advanced Delaunay triangulation algorithms. Possible extensions include Lawson edge-flipping methods, incremental Bowyer-Watson triangulation, constrained Delaunay triangulations, adaptive exact predicates, and higher-dimensional insphere tests. More sophisticated implementations may also incorporate robust arithmetic, hierarchical spatial search structures, and parallel retriangulation strategies to support large-scale mesh generation, interpolation, and finite-element discretization. Because local edge flips form one of the fundamental primitives of Delaunay algorithms, the implementation provides a natural bridge between determinant predicates and full triangulation construction methods used throughout scientific computing.

## 21.5.3. Incremental Construction and Bowyer-Watson Retriangulation

One of the most important Delaunay construction strategies is incremental insertion. Points are inserted one at a time into an existing triangulation. After inserting a new point $p$, the algorithm modifies the affected region so that the Delaunay condition is restored. A common formulation is the Bowyer-Watson algorithm.

Suppose $\mathcal{T}$ is a Delaunay triangulation of the already inserted points. When a new point $p$ is added, the algorithm identifies all existing triangles whose circumcircles contain $p$:

$$
C(p)
=
\left\{
T \in \mathcal{T} : p \text{ lies inside the circumcircle of } T
\right\}
\tag{21.5.20}
$$

This set is called the cavity. In determinant form, if $T=(a,b,c)$, membership in the cavity is determined by:

$$
\operatorname{incircle}(a,b,c,p)
\ \text{has the interior sign}
\tag{21.5.21}
$$

The triangles in $\mathcal{C}(p)$ are removed. The boundary of the cavity is then connected to the new point $p$, producing new triangles:

$$
(e_1,e_2)\subset \partial C(p)
\quad\Longrightarrow\quad
(e_1,e_2,p)\in \mathcal{T}_{\mathrm{new}}
\tag{21.5.22}
$$

with orientations chosen consistently.

The correctness of this procedure relies on the fact that the cavity boundary consists of edges separating deleted triangles from retained triangles. Thus, the algorithm must distinguish interior cavity edges from boundary cavity edges. An edge is internal to the cavity if it is shared by two removed triangles. It lies on the cavity boundary if it is incident to exactly one removed triangle. This is a topological condition, not merely a geometric one, and it illustrates why Delaunay algorithms must maintain connectivity carefully.

In two dimensions, incremental Delaunay triangulation is often presented as having expected complexity,

$$O(n\log n)\tag{21.5.23}$$

under standard assumptions and suitable point-location strategies. Without efficient point location, locating the triangle that contains each inserted point can dominate the cost. A naive search over all triangles is too expensive for large data sets. Practical algorithms therefore combine insertion with spatial search, walking methods, hierarchical triangulations, grids, or tree-based accelerators.

The Bowyer-Watson method generalizes to three dimensions by replacing triangles with tetrahedra and circumcircles with circumspheres. The cavity becomes the set of tetrahedra whose circumspheres contain the inserted point:

$$
\mathcal{C}(p)
=
\left\{
K \in \mathcal{T} :
p\ \text{lies inside the circumsphere of } K
\right\}
\tag{21.5.24}
$$

The boundary of this cavity is a triangular surface, and new tetrahedra are formed by connecting $p$ to each boundary face. The same conceptual algorithm applies, but the implementation is substantially more difficult because the cavity boundary must be a valid closed surface, orientation must be consistent, and degenerate cospherical configurations must be handled reliably.

Parallel and GPU implementations require additional redesign. Delaunay insertion is naturally local but can create conflicts when nearby points are inserted simultaneously. Modern GPU-parallel three-dimensional Delaunay methods therefore focus on locality, load balancing, conflict resolution, and post-processing rather than simply translating a serial insertion algorithm to parallel hardware. This is why current Delaunay research continues to be relevant for scientific computing at large scale (Gao and Chen, 2025).

### Rust Implementation

Following the discussion in Section 21.5.3 on incremental Delaunay construction, cavity retriangulation, and Bowyer-Watson insertion strategies, Program 21.5.3 provides a practical implementation of incremental Delaunay triangulation in two dimensions using determinant-based circumcircle predicates and local cavity reconstruction. In computational geometry, incremental insertion methods are among the most important approaches for constructing Delaunay triangulations because they transform a global triangulation problem into a sequence of local topological updates. The implementation demonstrates how the cavity definition introduced in Equations (21.5.20)–(21.5.22) can be realized algorithmically through repeated identification of triangles whose circumcircles contain the inserted point, followed by retriangulation of the resulting cavity boundary. The program also illustrates the topological distinction between interior cavity edges and cavity-boundary edges, which is essential for constructing valid Delaunay triangulations during insertion. Particular emphasis is placed on orientation consistency, supertriangle initialization, cavity-boundary extraction, and empty-circumcircle validation because reliable connectivity maintenance is central to practical Delaunay algorithms.

At the core of the implementation is the `Point2` structure, which represents planar geometric coordinates and provides the coordinate abstraction used throughout the triangulation process. Each point stores Cartesian coordinates `(x,y)` and supports the geometric operations required for orientation and circumcircle evaluation. The `Edge` structure represents an undirected triangulation edge through a pair of vertex indices stored in canonical order. By enforcing a consistent ordering of edge indices, the implementation ensures that cavity-boundary detection can be performed reliably using edge multiplicities. The `Triangle` structure stores the three vertex indices defining an oriented planar triangle and additionally provides utility methods for extracting triangle edges and detecting whether a triangle contains vertices belonging to the artificial supertriangle.

The `orient2d` function implements the determinant-based orientation predicate used throughout the triangulation algorithm. Given three planar points, the function evaluates the signed area determinant that determines whether the ordered triple is counterclockwise, clockwise, or degenerate. This orientation predicate is essential because all Delaunay predicates depend on consistent orientation conventions. The `ensure_ccw` function uses this orientation test to guarantee that every generated triangle maintains positive orientation. If a triangle is detected to have negative orientation, the function swaps two vertices to restore a counterclockwise ordering. This ensures that subsequent incircle evaluations have consistent determinant signs.

The `incircle_raw` function implements the translated determinant form of the incircle predicate described in Equation (21.5.21). By translating coordinates relative to the query point, the determinant evaluates whether a point lies inside, outside, or exactly on the circumcircle of a triangle. The `oriented_incircle` function then adjusts the determinant sign according to the orientation of the reference triangle so that the resulting predicate becomes orientation independent. The `point_inside_circumcircle` function converts this determinant evaluation into a logical geometric test by determining whether the inserted point lies strictly inside the circumcircle of a triangle. Together, these functions implement the geometric criterion defining cavity membership in Equation (21.5.20).

The `bounding_box` and `add_supertriangle` functions initialize the triangulation process. The bounding-box computation determines an enclosing rectangular region for the input point set, while the supertriangle construction generates a sufficiently large enclosing triangle containing all inserted points. This artificial supertriangle provides a valid initial triangulation before point insertion begins. After all insertions are complete, triangles connected to supertriangle vertices are removed to recover the final Delaunay triangulation of the original point set.

The `cavity_indices` function identifies the Bowyer-Watson cavity associated with an inserted point. For every active triangle in the triangulation, the function evaluates whether the inserted point lies inside the triangle’s circumcircle. If so, the triangle is added to the cavity set described in Equation (21.5.20). This cavity represents the local region where the Delaunay condition is violated by the newly inserted point and therefore must be retriangulated.

The `boundary_edges_of_cavity` function extracts the boundary of the cavity through edge multiplicity analysis. Every edge belonging to a removed triangle is counted using a hash table. Edges shared by two removed triangles are interior cavity edges and therefore disappear during retriangulation, while edges appearing exactly once form the cavity boundary separating removed triangles from retained triangles. This realizes the topological distinction described in Section 21.5.3 between internal cavity edges and boundary cavity edges. The `remove_cavity_triangles` function then deletes all cavity triangles from the active triangulation.

The `bowyer_watson` function orchestrates the complete incremental Delaunay insertion process. Beginning from the supertriangle initialization, the algorithm inserts points sequentially. For each inserted point, the cavity is identified, boundary edges are extracted, cavity triangles are removed, and new triangles are generated by connecting the inserted point to each cavity-boundary edge as described in Equation (21.5.22). Newly generated triangles are then reoriented consistently before being inserted into the active triangulation. The function also prints insertion diagnostics reporting cavity size, boundary size, and the number of active triangles after each insertion step. Finally, all triangles incident to supertriangle vertices are removed to produce the final Delaunay triangulation.

The `triangle_area` function computes the area of a triangle using the determinant-based orientation formula. The `circumcircle_contains_any_other_point` and `validate_delaunay` functions provide a direct verification of the empty-circumcircle property for the final triangulation. For each triangle, the validation routine checks whether any nonincident point lies strictly inside the circumcircle. If no such point exists for every triangle, the triangulation satisfies the Delaunay condition globally.

The `print_points`, `print_triangles`, and `print_edges` utility functions provide structured diagnostic output for the generated triangulation. The edge-reporting function additionally classifies edges as boundary or interior according to their multiplicities within the triangulation. Such connectivity diagnostics are particularly important in Delaunay implementations because triangulation correctness depends simultaneously on geometric predicates and topological consistency.

The `main` function serves to demonstrate the complete Bowyer-Watson insertion process for a representative two-dimensional point cloud. It begins by defining a scattered planar point set and initializing the geometric tolerance used for determinant evaluation. The program then constructs the Delaunay triangulation incrementally through repeated cavity detection and retriangulation. After all insertions are completed, the implementation prints the final triangulation, reports edge multiplicities, and verifies that the empty-circumcircle condition is satisfied globally. The resulting output demonstrates that local cavity retriangulation successfully restores the Delaunay condition after each insertion while preserving consistent triangulation connectivity. Altogether, the implementation illustrates how local determinant predicates and topological cavity reconstruction combine to produce a complete incremental Delaunay triangulation algorithm suitable for scientific-computing applications.

```rust
// Program 21.5.3: Incremental Bowyer-Watson Delaunay Triangulation in Two Dimensions
//
// Problem statement:
// Given a finite set of planar points, construct a two-dimensional Delaunay
// triangulation using incremental insertion and Bowyer-Watson retriangulation.
// For each inserted point, the algorithm identifies the cavity of triangles
// whose circumcircles contain the point, removes those triangles, extracts
// the boundary edges of the cavity, and connects each boundary edge to the
// inserted point with consistently oriented triangles.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Edge {
    u: usize,
    v: usize,
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    a: usize,
    b: usize,
    c: usize,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl Edge {
    fn new(u: usize, v: usize) -> Self {
        if u < v {
            Self { u, v }
        } else {
            Self { u: v, v: u }
        }
    }
}

impl Triangle {
    fn edges(self) -> [Edge; 3] {
        [
            Edge::new(self.a, self.b),
            Edge::new(self.b, self.c),
            Edge::new(self.c, self.a),
        ]
    }

    fn contains_super_vertex(self, original_count: usize) -> bool {
        self.a >= original_count || self.b >= original_count || self.c >= original_count
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn ensure_ccw(t: Triangle, points: &[Point2]) -> Triangle {
    if orient2d(points[t.a], points[t.b], points[t.c]) >= 0.0 {
        t
    } else {
        Triangle {
            a: t.a,
            b: t.c,
            c: t.b,
        }
    }
}

fn incircle_raw(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let ax = a.x - d.x;
    let ay = a.y - d.y;
    let bx = b.x - d.x;
    let by = b.y - d.y;
    let cx = c.x - d.x;
    let cy = c.y - d.y;

    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;

    ax * (by * c2 - b2 * cy)
        - ay * (bx * c2 - b2 * cx)
        + a2 * (bx * cy - by * cx)
}

fn oriented_incircle(a: Point2, b: Point2, c: Point2, d: Point2) -> f64 {
    let det = incircle_raw(a, b, c, d);

    if orient2d(a, b, c) > 0.0 {
        det
    } else {
        -det
    }
}

fn point_inside_circumcircle(a: Point2, b: Point2, c: Point2, p: Point2, eps: f64) -> bool {
    oriented_incircle(a, b, c, p) > eps
}

fn bounding_box(points: &[Point2]) -> (f64, f64, f64, f64) {
    let mut xmin = points[0].x;
    let mut xmax = points[0].x;
    let mut ymin = points[0].y;
    let mut ymax = points[0].y;

    for &p in points {
        xmin = xmin.min(p.x);
        xmax = xmax.max(p.x);
        ymin = ymin.min(p.y);
        ymax = ymax.max(p.y);
    }

    (xmin, xmax, ymin, ymax)
}

fn add_supertriangle(points: &mut Vec<Point2>, original_count: usize) -> Triangle {
    let (xmin, xmax, ymin, ymax) = bounding_box(&points[..original_count]);

    let dx = xmax - xmin;
    let dy = ymax - ymin;
    let delta = dx.max(dy).max(1.0);
    let cx = 0.5 * (xmin + xmax);
    let cy = 0.5 * (ymin + ymax);

    let p0 = Point2::new(cx - 20.0 * delta, cy - delta);
    let p1 = Point2::new(cx, cy + 20.0 * delta);
    let p2 = Point2::new(cx + 20.0 * delta, cy - delta);

    let i0 = points.len();
    let i1 = points.len() + 1;
    let i2 = points.len() + 2;

    points.push(p0);
    points.push(p1);
    points.push(p2);

    ensure_ccw(
        Triangle {
            a: i0,
            b: i1,
            c: i2,
        },
        points,
    )
}

fn cavity_indices(
    triangles: &[Triangle],
    points: &[Point2],
    inserted_point: Point2,
    eps: f64,
) -> Vec<usize> {
    let mut cavity = Vec::new();

    for (i, &t) in triangles.iter().enumerate() {
        if point_inside_circumcircle(
            points[t.a],
            points[t.b],
            points[t.c],
            inserted_point,
            eps,
        ) {
            cavity.push(i);
        }
    }

    cavity
}

fn boundary_edges_of_cavity(triangles: &[Triangle], cavity: &[usize]) -> Vec<Edge> {
    let mut counts: HashMap<Edge, usize> = HashMap::new();

    for &idx in cavity {
        for e in triangles[idx].edges() {
            *counts.entry(e).or_insert(0) += 1;
        }
    }

    let mut boundary = Vec::new();

    for (edge, count) in counts {
        if count == 1 {
            boundary.push(edge);
        }
    }

    boundary
}

fn remove_cavity_triangles(triangles: &[Triangle], cavity: &[usize]) -> Vec<Triangle> {
    let mut remove = vec![false; triangles.len()];

    for &idx in cavity {
        remove[idx] = true;
    }

    triangles
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| if remove[i] { None } else { Some(t) })
        .collect()
}

fn bowyer_watson(input_points: &[Point2], eps: f64) -> (Vec<Point2>, Vec<Triangle>) {
    let original_count = input_points.len();

    let mut points = input_points.to_vec();
    let supertriangle = add_supertriangle(&mut points, original_count);

    let mut triangles = vec![supertriangle];

    for i in 0..original_count {
        let p = points[i];

        let cavity = cavity_indices(&triangles, &points, p, eps);
        let boundary = boundary_edges_of_cavity(&triangles, &cavity);

        let mut retained = remove_cavity_triangles(&triangles, &cavity);

        for &edge in &boundary {
            let candidate = Triangle {
                a: edge.u,
                b: edge.v,
                c: i,
            };

            let oriented = ensure_ccw(candidate, &points);

            if orient2d(points[oriented.a], points[oriented.b], points[oriented.c]).abs() > eps {
                retained.push(oriented);
            }
        }

        triangles = retained;

        println!(
            "insert p_{:<2}: cavity triangles = {:<3} boundary edges = {:<3} active triangles = {}",
            i,
            cavity.len(),
            boundary.len(),
            triangles.len()
        );
    }

    let final_triangles: Vec<Triangle> = triangles
        .into_iter()
        .filter(|t| !t.contains_super_vertex(original_count))
        .collect();

    (points[..original_count].to_vec(), final_triangles)
}

fn triangle_area(points: &[Point2], t: Triangle) -> f64 {
    0.5 * orient2d(points[t.a], points[t.b], points[t.c]).abs()
}

fn circumcircle_contains_any_other_point(points: &[Point2], t: Triangle, eps: f64) -> bool {
    for (i, &p) in points.iter().enumerate() {
        if i == t.a || i == t.b || i == t.c {
            continue;
        }

        if point_inside_circumcircle(points[t.a], points[t.b], points[t.c], p, eps) {
            return true;
        }
    }

    false
}

fn validate_delaunay(points: &[Point2], triangles: &[Triangle], eps: f64) -> bool {
    for &t in triangles {
        if circumcircle_contains_any_other_point(points, t, eps) {
            return false;
        }
    }

    true
}

fn print_points(points: &[Point2]) {
    println!("Input Points");
    println!("------------");

    for (i, p) in points.iter().enumerate() {
        println!("p_{:<2} = ({:>8.4}, {:>8.4})", i, p.x, p.y);
    }

    println!();
}

fn print_triangles(points: &[Point2], triangles: &[Triangle]) {
    println!("Final Delaunay Triangles");
    println!("------------------------");

    for (i, &t) in triangles.iter().enumerate() {
        let area = triangle_area(points, t);
        println!(
            "T_{:<2} = ({:>2}, {:>2}, {:>2})   area = {:>.10}",
            i, t.a, t.b, t.c, area
        );
    }

    println!("number of triangles = {}\n", triangles.len());
}

fn print_edges(triangles: &[Triangle]) {
    let mut edge_counts: HashMap<Edge, usize> = HashMap::new();

    for &t in triangles {
        for e in t.edges() {
            *edge_counts.entry(e).or_insert(0) += 1;
        }
    }

    let mut edges: Vec<(Edge, usize)> = edge_counts.into_iter().collect();
    edges.sort_by_key(|(e, _)| (e.u, e.v));

    println!("Triangulation Edges");
    println!("-------------------");

    for (e, count) in edges {
        let kind = if count == 1 { "boundary" } else { "interior" };
        println!(
            "edge ({:>2}, {:>2})   multiplicity = {}   {}",
            e.u, e.v, count, kind
        );
    }

    println!();
}

fn main() {
    let points = vec![
        Point2::new(0.10, 0.15),
        Point2::new(0.85, 0.10),
        Point2::new(0.95, 0.80),
        Point2::new(0.15, 0.90),
        Point2::new(0.50, 0.45),
        Point2::new(0.62, 0.70),
        Point2::new(0.30, 0.55),
        Point2::new(0.72, 0.35),
    ];

    let eps = 1.0e-12;

    println!("Incremental Bowyer-Watson Delaunay Triangulation");
    println!("================================================\n");

    print_points(&points);

    println!("Insertion Diagnostics");
    println!("---------------------");

    let (sites, triangles) = bowyer_watson(&points, eps);

    println!();

    print_triangles(&sites, &triangles);
    print_edges(&triangles);

    println!("Delaunay Validation");
    println!("-------------------");
    println!(
        "empty-circumcircle condition satisfied = {}",
        validate_delaunay(&sites, &triangles, eps)
    );
}
```

Program 21.5.3 demonstrates a practical implementation of incremental Delaunay triangulation using Bowyer-Watson cavity retriangulation. The implementation reflects one of the central ideas developed in Section 21.5.3: global Delaunay structure can be maintained through purely local geometric and topological updates. Instead of reconstructing an entire triangulation after each insertion, the algorithm identifies only the local cavity affected by the inserted point and retriangulates that region while leaving the remainder of the triangulation unchanged.

The numerical example illustrates several important geometric and topological behaviors. Each inserted point generates a cavity consisting of triangles whose circumcircles contain the new point, the cavity boundary is identified through edge multiplicity analysis, and retriangulation restores the empty-circumcircle condition locally. The final validation step confirms that the resulting triangulation satisfies the Delaunay property globally. The edge-multiplicity diagnostics additionally demonstrate the distinction between interior triangulation edges and boundary edges, highlighting the importance of connectivity maintenance during incremental insertion.

The modular structure of the implementation allows the framework to be extended naturally toward more advanced Delaunay triangulation algorithms. Possible extensions include point-location acceleration structures, constrained Delaunay triangulations, adaptive exact predicates, weighted triangulations, parallel insertion strategies, and higher-dimensional Bowyer-Watson tetrahedralization. More sophisticated implementations may also incorporate hierarchical search structures, conflict-resolution methods for GPU insertion, and robust symbolic perturbation techniques for handling degenerate cocircular configurations. Because incremental cavity retriangulation forms one of the fundamental paradigms of modern mesh generation, the implementation provides a natural bridge between determinant predicates, triangulation connectivity, and large-scale geometric discretization methods used throughout scientific computing.

## 21.5.4. Constrained, Weighted, and Scientific-Computing Variants

The unconstrained Delaunay triangulation of a point set fills the convex hull of the points. Many scientific domains, however, are not convex and contain prescribed boundaries, holes, interfaces, or material regions. If a domain boundary contains a segment, polygonal chain, or surface that must appear in the mesh, the triangulation must be constrained. In two dimensions, a constrained Delaunay triangulation requires specified input edges to be present:

$$E_{\mathrm{boundary}}\subset E_{\mathcal{T}} \tag{21.5.25}$$

The Delaunay condition is then modified so that visibility across constrained edges is not required. In other words, a triangle may be considered valid if its circumcircle contains no visible vertex that violates the constrained Delaunay criterion. This is essential for meshing polygonal domains whose boundary cannot be replaced by arbitrary Delaunay edges.

Conforming Delaunay methods take a different approach. Instead of merely forcing boundary edges into the triangulation, they add Steiner points so that the final triangulation satisfies both boundary conformity and stronger quality or Delaunay properties. If $P$ is the original point set and $S$ is a set of inserted Steiner points, the final triangulation is built on:

$$P^\ast = P\cup S \tag{21.5.26}$$

The purpose of $S$ is to improve element quality, recover boundaries, or satisfy mesh-size constraints. This idea connects Delaunay triangulation directly to mesh generation, where the input is not just a point set but a geometric domain with physical and numerical requirements.

Weighted Delaunay triangulations and power diagrams generalize ordinary Voronoi diagrams by assigning a weight $w_i\in\mathbb{R}$ to each site $p_i$. The power distance from $x$ to the weighted site $(p_i,w_i)$ is:

$$\pi_i(x) = \|x-p_i\|_2^2-w_i \tag{21.5.27}$$

The power cell of site $i$ is,

$$V_i^{\mathrm{pow}} = \{x\in\mathbb{R}^d:\pi_i(x)\le \pi_j(x)\ \text{for all }j\} \tag{21.5.28}$$

The boundary between two weighted cells is still a hyperplane because,

$$\|x-p_i\|_2^2-w_i\le\|x-p_j\|_2^2-w_j\tag{21.5.29}$$

reduces to a linear inequality in $x$:

$$2(p_j-p_i)\cdot x\le\|p_j\|_2^2-\|p_i\|_2^2+w_i-w_j \tag{21.5.30}$$

Power diagrams are important in optimization, anisotropic modeling, particle methods, and semi-discrete optimal transport. In semi-discrete optimal transport, one often seeks weights such that weighted Voronoi cells have prescribed measures:

$$|V_i^{\mathrm{pow}}| = m_i,\qquad i=1,\ldots,n \tag{21.5.31}$$

for given masses $m_i>0$. Large-scale distributed Voronoi computation shows that these weighted cell structures are now practical scientific-computing primitives rather than only theoretical geometric constructions (Lévy et al., 2025).

Delaunay and Voronoi structures also support interpolation. Given a scattered point set, Delaunay triangles or tetrahedra provide local simplices on which barycentric interpolation can be applied:

$$
Iu(x)

\sum_{i=0}^d \lambda_i(x)u(v_i),\qquad x\in \operatorname{conv}{v_0,\ldots,v_d}\tag{21.5.32}
$$

Voronoi cells, by contrast, naturally support nearest-neighbor interpolation and control-volume discretizations. If $x\in V_i$, then the nearest-neighbor interpolant is:

$$I_{\mathrm{NN}}u(x)=u(p_i) \tag{21.5.33}$$

Thus, Delaunay and Voronoi constructions provide complementary discretization viewpoints: one simplex-based and interpolation-oriented, the other cell-based and proximity-oriented.

### Rust Implementation

Following the discussion in Section 21.5.4 on constrained triangulations, weighted Voronoi structures, power diagrams, and interpolation-oriented geometric discretizations, Program 21.5.4 provides a practical implementation of weighted Voronoi-cell construction and power-diagram computation in two dimensions. In computational geometry and scientific computing, weighted Voronoi structures generalize ordinary nearest-neighbor geometry by incorporating scalar weights into the distance metric, thereby allowing regions of influence to reflect heterogeneous physical, geometric, or optimization-based properties. The implementation demonstrates how the weighted half-space inequalities introduced in Equations (21.5.27)–(21.5.30) can be realized computationally through iterative polygon clipping and power-distance evaluation. The program additionally illustrates the complementary relationship between Voronoi-cell discretizations and simplex-based interpolation by combining weighted power-cell construction with barycentric interpolation on Delaunay simplices. Particular emphasis is placed on half-space geometry, power-distance classification, convex polygon clipping, and interpolation consistency because these ideas form the geometric foundation of many modern scientific discretization methods.

At the core of the implementation is the `Point2` structure, which represents planar geometric coordinates and provides the vector operations required throughout the power-diagram construction process. In addition to storing Cartesian coordinates `(x,y)`, the structure implements dot products, squared Euclidean norms, and squared-distance evaluations. These operations are used repeatedly in weighted half-space generation, power-distance evaluation, polygon clipping, and barycentric interpolation. The abstraction ensures that all geometric computations remain consistent and reusable across the different stages of the algorithm.

The `WeightedSite` structure extends ordinary geometric sites by associating a scalar weight with each planar point. This directly realizes the weighted-site formulation introduced in Equation (21.5.27). By storing both the geometric location and the associated weight, the structure allows the implementation to represent weighted Voronoi sites whose regions of influence are determined not purely by Euclidean proximity but by power distance. The `HalfSpace` structure stores the coefficients `(a,b,c)` defining the linear inequalities associated with weighted Voronoi boundaries, while the `Triangle` structure stores oriented simplices used later for barycentric interpolation.

The `orient2d` function implements the determinant-based orientation predicate used throughout the program. Given three planar points, the function evaluates the signed area determinant that determines whether the ordered triple is counterclockwise, clockwise, or degenerate. This predicate is essential for polygon orientation consistency and for computing barycentric coordinates during simplex interpolation.

The `weighted_half_space` function implements the weighted Voronoi inequality derived from Equations (21.5.29)–(21.5.30). Given two weighted sites `(p_i,w_i)` and `(p_j,w_j)`, the function constructs the linear half-space describing the set of points whose power distance to `p_i` does not exceed the power distance to `p_j`. Although the original weighted-distance expression is quadratic, the difference of squared distances reduces to a linear inequality in the spatial coordinates. This property is fundamental because it guarantees that weighted power cells remain convex polyhedral regions.

The `satisfies_half_space` and `line_intersection` functions provide the geometric primitives required for polygon clipping. The first evaluates whether a point satisfies a weighted half-space inequality within a numerical tolerance, while the second computes the intersection point between a polygon edge and a clipping boundary. These operations form the numerical basis for constructing weighted Voronoi cells through repeated half-space intersections.

The `clip_polygon_by_half_space` function implements iterative polygon clipping using a Sutherland-Hodgman-style procedure. Each polygon edge is classified relative to the clipping half-space, and intersection points are inserted whenever an edge crosses the boundary. Through repeated clipping against all competing weighted-site inequalities, the algorithm constructs the bounded power cell associated with a particular weighted site. Because power cells are convex intersections of linear half-spaces, this clipping strategy provides a direct computational realization of Equation (21.5.28).

The `power_cell` function orchestrates the complete weighted-cell construction process. Beginning from an initial rectangular bounding box, the function clips the polygon successively against the weighted half-spaces generated by all competing sites. Each clipping step reduces the feasible region until the remaining polygon corresponds to the bounded weighted Voronoi cell of the selected site. This procedure provides a geometric implementation of weighted Voronoi partitioning entirely through convex half-space intersection.

The `polygon_area` function evaluates the area of a polygon using the standard shoelace summation formula. Since the weighted cells partition the bounding region, the total area of all cells should equal the area of the enclosing box. The area computation therefore serves both as a geometric diagnostic and as a verification of clipping consistency.

The `power_distance` and `nearest_power_site` functions implement the weighted nearest-neighbor geometry associated with power diagrams. The first evaluates the weighted distance introduced in Equation (21.5.27), while the second determines which weighted site minimizes this quantity for a given query point. Unlike ordinary nearest-neighbor classification, the resulting partition depends not only on geometric proximity but also on the associated weights, allowing weighted cells to expand or contract according to the assigned scalar parameters.

The `barycentric_coordinates` function computes barycentric coordinates inside a planar simplex using orientation determinants. Given a query point and a triangle, the function evaluates the barycentric weights associated with the simplex vertices. These coordinates provide the basis for simplex-based interpolation because they express the query point as an affine combination of the triangle vertices. The `barycentric_interpolation` function then applies these coordinates to interpolate scalar nodal values according to the simplex-based interpolation formula described in Equation (21.5.32).

The `print_sites` and `print_cell` utility functions provide structured diagnostic output for the weighted sites and the resulting power cells. In addition to printing polygonal vertices and cell areas, the implementation reports the weighted nearest-site classification associated with each vertex. Such diagnostics are important in computational geometry because weighted Voronoi partitions depend sensitively on both geometric configuration and assigned weights.

The `main` function serves to demonstrate the complete weighted power-diagram construction and interpolation framework for a representative two-dimensional weighted point set. It begins by defining several weighted sites together with a rectangular bounding region. The program then constructs each weighted Voronoi cell through repeated half-space clipping, computes the associated polygonal areas, and verifies that the resulting cells partition the bounding box consistently. Next, the implementation demonstrates barycentric interpolation inside a Delaunay simplex by computing barycentric coordinates and evaluating an interpolated scalar field. Finally, the program compares this simplex-based interpolation with weighted nearest-neighbor interpolation using the power-diagram classification. Altogether, the implementation illustrates the complementary relationship between Voronoi-cell discretizations and simplex-based interpolation methods within scientific computing.

```rust
// Program 21.5.4: Weighted Voronoi Cells and Power-Diagram Construction
//
// Problem statement:
// Given a finite set of weighted planar sites (p_i,w_i), construct bounded
// power-diagram cells using weighted half-space intersections. The program
// demonstrates weighted Voronoi geometry, power-distance evaluation,
// nearest-power-site classification, and barycentric interpolation on
// Delaunay simplices.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct WeightedSite {
    position: Point2,
    weight: f64,
}

#[derive(Clone, Copy, Debug)]
struct HalfSpace {
    a: f64,
    b: f64,
    c: f64,
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    a: Point2,
    b: Point2,
    c: Point2,
}

impl Point2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn dot(self, other: Point2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    fn distance_squared(self, other: Point2) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

impl WeightedSite {
    fn new(x: f64, y: f64, weight: f64) -> Self {
        Self {
            position: Point2::new(x, y),
            weight,
        }
    }
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn weighted_half_space(site_i: WeightedSite, site_j: WeightedSite) -> HalfSpace {
    let pi = site_i.position;
    let pj = site_j.position;

    HalfSpace {
        a: 2.0 * (pj.x - pi.x),
        b: 2.0 * (pj.y - pi.y),
        c: pj.norm_squared() - pi.norm_squared()
            + site_i.weight
            - site_j.weight,
    }
}

fn satisfies_half_space(p: Point2, h: HalfSpace, eps: f64) -> bool {
    h.a * p.x + h.b * p.y <= h.c + eps
}

fn line_intersection(p: Point2, q: Point2, h: HalfSpace) -> Point2 {
    let fp = h.a * p.x + h.b * p.y - h.c;
    let fq = h.a * q.x + h.b * q.y - h.c;

    let t = fp / (fp - fq);

    Point2::new(
        p.x + t * (q.x - p.x),
        p.y + t * (q.y - p.y),
    )
}

fn clip_polygon_by_half_space(
    polygon: &[Point2],
    h: HalfSpace,
    eps: f64,
) -> Vec<Point2> {
    if polygon.is_empty() {
        return Vec::new();
    }

    let mut clipped = Vec::new();

    for i in 0..polygon.len() {
        let current = polygon[i];
        let next = polygon[(i + 1) % polygon.len()];

        let current_inside = satisfies_half_space(current, h, eps);
        let next_inside = satisfies_half_space(next, h, eps);

        match (current_inside, next_inside) {
            (true, true) => {
                clipped.push(next);
            }
            (true, false) => {
                clipped.push(line_intersection(current, next, h));
            }
            (false, true) => {
                clipped.push(line_intersection(current, next, h));
                clipped.push(next);
            }
            (false, false) => {}
        }
    }

    clipped
}

fn power_cell(
    sites: &[WeightedSite],
    site_index: usize,
    bounding_box: &[Point2],
    eps: f64,
) -> Vec<Point2> {
    let mut cell = bounding_box.to_vec();

    let site_i = sites[site_index];

    for (j, &site_j) in sites.iter().enumerate() {
        if j == site_index {
            continue;
        }

        let h = weighted_half_space(site_i, site_j);

        cell = clip_polygon_by_half_space(&cell, h, eps);

        if cell.is_empty() {
            break;
        }
    }

    cell
}

fn polygon_area(poly: &[Point2]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }

    let mut sum = 0.0;

    for i in 0..poly.len() {
        let p = poly[i];
        let q = poly[(i + 1) % poly.len()];

        sum += p.x * q.y - p.y * q.x;
    }

    0.5 * sum.abs()
}

fn power_distance(x: Point2, site: WeightedSite) -> f64 {
    x.distance_squared(site.position) - site.weight
}

fn nearest_power_site(
    x: Point2,
    sites: &[WeightedSite],
) -> usize {
    let mut best = 0usize;
    let mut best_value = power_distance(x, sites[0]);

    for i in 1..sites.len() {
        let value = power_distance(x, sites[i]);

        if value < best_value {
            best_value = value;
            best = i;
        }
    }

    best
}

fn barycentric_coordinates(
    p: Point2,
    t: Triangle,
) -> Option<(f64, f64, f64)> {
    let det = orient2d(t.a, t.b, t.c);

    if det.abs() < 1.0e-14 {
        return None;
    }

    let lambda0 = orient2d(p, t.b, t.c) / det;
    let lambda1 = orient2d(t.a, p, t.c) / det;
    let lambda2 = orient2d(t.a, t.b, p) / det;

    Some((lambda0, lambda1, lambda2))
}

fn barycentric_interpolation(
    p: Point2,
    t: Triangle,
    values: [f64; 3],
) -> Option<f64> {
    barycentric_coordinates(p, t).map(|(l0, l1, l2)| {
        l0 * values[0]
            + l1 * values[1]
            + l2 * values[2]
    })
}

fn print_sites(sites: &[WeightedSite]) {
    println!("Weighted Sites");
    println!("--------------");

    for (i, s) in sites.iter().enumerate() {
        println!(
            "p_{:<2} = ({:>8.4}, {:>8.4}), weight = {:>.4}",
            i,
            s.position.x,
            s.position.y,
            s.weight
        );
    }

    println!();
}

fn print_cell(
    index: usize,
    cell: &[Point2],
    sites: &[WeightedSite],
) {
    println!("Power Cell V_pow_{}", index);
    println!("-------------------");

    if cell.is_empty() {
        println!("cell is empty\n");
        return;
    }

    for (k, v) in cell.iter().enumerate() {
        let nearest = nearest_power_site(*v, sites);

        println!(
            "v_{:<2} = ({:>10.6}, {:>10.6})   nearest weighted site = p_{}",
            k,
            v.x,
            v.y,
            nearest
        );
    }

    println!("number of vertices = {}", cell.len());
    println!("bounded cell area  = {:>.10}\n", polygon_area(cell));
}

fn main() {
    let sites = vec![
        WeightedSite::new(0.20, 0.20, 0.02),
        WeightedSite::new(0.80, 0.25, 0.10),
        WeightedSite::new(0.65, 0.80, 0.04),
        WeightedSite::new(0.25, 0.75, 0.00),
    ];

    let bounding_box = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(1.0, 1.0),
        Point2::new(0.0, 1.0),
    ];

    let eps = 1.0e-12;

    println!("Weighted Voronoi Cells and Power-Diagram Construction");
    println!("=====================================================\n");

    print_sites(&sites);

    println!("Bounding Box");
    println!("------------");

    for (i, p) in bounding_box.iter().enumerate() {
        println!(
            "b_{:<2} = ({:>8.4}, {:>8.4})",
            i,
            p.x,
            p.y
        );
    }

    println!();

    let mut total_area = 0.0;

    for i in 0..sites.len() {
        let cell = power_cell(&sites, i, &bounding_box, eps);

        total_area += polygon_area(&cell);

        print_cell(i, &cell, &sites);
    }

    println!("Partition Check");
    println!("---------------");
    println!("sum of bounded cell areas = {:>.10}", total_area);
    println!("bounding-box area         = {:>.10}", 1.0);
    println!(
        "absolute difference       = {:>.3e}",
        (total_area - 1.0_f64).abs()
    );

    println!();

    let triangle = Triangle {
        a: sites[0].position,
        b: sites[1].position,
        c: sites[2].position,
    };

    let nodal_values = [1.0, 2.0, 1.5];

    let x = Point2::new(0.55, 0.40);

    println!("Barycentric Interpolation");
    println!("-------------------------");

    if let Some((l0, l1, l2)) =
        barycentric_coordinates(x, triangle)
    {
        println!(
            "lambda = ({:>.6}, {:>.6}, {:>.6})",
            l0,
            l1,
            l2
        );

        if let Some(value) =
            barycentric_interpolation(x, triangle, nodal_values)
        {
            println!(
                "interpolated value at x = {:>.10}",
                value
            );
        }
    } else {
        println!("degenerate triangle");
    }

    println!();

    println!("Nearest-Neighbor Interpolation");
    println!("------------------------------");

    let nearest = nearest_power_site(x, &sites);

    println!(
        "x belongs to weighted cell of p_{}",
        nearest
    );

    println!(
        "nearest-neighbor interpolant value = {:>.10}",
        nodal_values[nearest.min(2)]
    );
}
```

Program 21.5.4 demonstrates a practical implementation of weighted Voronoi geometry, power-diagram construction, and interpolation-oriented geometric discretization. The implementation reflects one of the central ideas developed in Section 21.5.4: geometric partitions and interpolation structures can be generalized naturally through weighted distance functions while still preserving convexity and computational tractability.

The numerical example illustrates several important geometric behaviors. Weighted half-space clipping successfully constructs convex power cells whose union partitions the bounding region exactly within floating-point precision. The resulting cells differ from ordinary Voronoi regions because the assigned weights modify the effective regions of influence of the generating sites. The barycentric interpolation example additionally demonstrates how Delaunay simplices support continuous interpolation, while the weighted nearest-neighbor construction illustrates the complementary cell-based discretization viewpoint associated with Voronoi structures.

The modular design of the implementation allows the framework to be extended naturally toward more advanced weighted geometric algorithms. Possible extensions include constrained power diagrams, anisotropic Voronoi structures, semi-discrete optimal transport solvers, adaptive mesh refinement, centroidal Voronoi tessellations, and distributed weighted-cell computation. More sophisticated implementations may also incorporate robust exact predicates, parallel polygon clipping, GPU-based spatial partitioning, and higher-dimensional weighted Delaunay structures. Because weighted Voronoi and Delaunay constructions provide a bridge between geometric partitioning, interpolation, and optimization, the implementation forms a natural foundation for many modern scientific-computing discretization methods.

## 21.5.5. Robustness, Complexity, and Rust Implementation Notes

The mathematical definitions of Delaunay and Voronoi structures are exact, but their implementation depends on finite-precision predicates. The two most important signs are orientation and incircle or insphere:

$$\operatorname{orient2d},\quad\operatorname{orient3d},\quad\operatorname{incircle},\quad\operatorname{insphere} \tag{21.5.34}$$

These predicates determine whether cells are consistently oriented, whether points lie inside circumcircles or circumspheres, whether edges should be flipped, and whether cavities should be removed during insertion. Near cocircular or cospherical configurations, the determinant values may be close to zero:

$$\operatorname{incircle}(a,b,c,d)\approx 0,\qquad\operatorname{insphere}(a,b,c,d,e)\approx 0 \tag{21.5.35}$$

Such cases require exact or filtered evaluation if the triangulation is to be combinatorially reliable. Tolerance-based decisions may be acceptable in exploratory or approximate settings, but they do not guarantee globally consistent triangulations.

The standard complexity statements should be interpreted with implementation context. In two dimensions, Delaunay triangulation is commonly achievable in $O(n\log n)$ time using divide-and-conquer, sweep, or randomized incremental methods with efficient point location. The Voronoi diagram has linear combinatorial complexity in the number of sites for nondegenerate planar inputs $O(n)$. In three dimensions, the worst-case complexity of Delaunay tetrahedralization can be higher because the number of tetrahedra may grow superlinearly. In practice, performance depends strongly on point distribution, degeneracy, data layout, and the cost of robust predicates.

For Rust implementation, Delaunay triangulation should be designed as a connectivity problem as much as a numeric one. A useful conceptual pipeline is

$$\text{points}\longrightarrow\text{predicate kernel}\longrightarrow\text{point location}\longrightarrow\text{local cavity or flip}\\ \longrightarrow\text{connectivity update}\longrightarrow\text{validation}\tag{21.5.36}$$

The predicate kernel supplies orientation and incircle or insphere signs. The point-location layer finds the simplex containing or near a new point. The local update layer performs edge flips or cavity retriangulation. The connectivity layer maintains stable adjacency information. The validation layer checks orientation, boundary consistency, and local Delaunay conditions.

The connectivity structure should use stable indices rather than pointer-heavy mutable references. A lightweight implementation may begin by storing only vertex indices for each simplex:

$$T_k=(i,j,\ell) \tag{21.5.37}$$

where $i,j,\ell$ index vertices in contiguous storage arrays. Edge relationships and cavity boundaries can then be reconstructed through edge multiplicities or adjacency searches, as demonstrated in the incremental Bowyer-Watson implementation. More advanced triangulation data structures additionally store explicit neighbor-triangle indices:

$$T_k=(i,j,\ell;\ n_0,n_1,n_2) \tag{21.5.38}$$

where $n_0$, $n_1$, $n_2$ denote adjacent triangles or boundary markers opposite the corresponding vertices. Such layouts support efficient point location, cache-aware traversal, local retriangulation, and safe mutation patterns. For tetrahedral meshes, the same idea extends naturally to four vertex indices together with four neighboring tetrahedra.

The main implementation warning is that Delaunay algorithms mix local numerical decisions with global topology. A wrong incircle sign can cause a wrong flip. A wrong cavity boundary can corrupt the triangulation. A missing adjacency update can invalidate later point location. Therefore, a robust implementation should not treat Delaunay triangulation as only a collection of determinant formulas. It should treat it as a coordinated geometric data structure whose correctness depends on certified predicates, consistent degeneracy handling, and carefully checked local connectivity updates.

# 21.6. Spatial Search and Nearest-Neighbor Data Structures

Spatial search is the algorithmic layer that turns geometric data into queryable scientific data. Once a code stores a point cloud, mesh, particle set, collection of bounding boxes, or unstructured cell complex, it must repeatedly answer questions of proximity and containment. Which point is closest to a query location? Which particles lie within a fixed interaction radius? Which mesh cell might contain a point? Which bounding boxes intersect a ray, segment, or control volume? These questions arise in interpolation, particle and meshless methods, point-cloud processing, collision and contact detection, nearest-cell lookup, conservative transfer, clustering, and scientific visualization. The essential issue is that direct search is often too expensive, so the data must be organized into a spatial index. Modern point-cloud and neighborhood-search literature emphasizes that the practical performance of such indexes depends not only on asymptotic complexity, but also on memory locality, dimensionality, query type, and data layout (Teuscher et al., 2025; Viñambres et al., 2026; Laso and Yermo, 2026; Ting et al., 2024; Benthin et al., 2024).

## 21.6.1. Query Types and the Brute-Force Baseline

Let

$$P=\{p_1,p_2,\ldots,p_N\}\subset \mathbb{R}^d\tag{21.6.1}$$

be a finite point set. The most basic spatial query is nearest-neighbor search. Given a query point $x\in\mathbb{R}^d$, the nearest neighbor is:

$$p_{j^\ast}\quad\text{where}\quad j^\ast=\arg\min_{1\le j\le N}\|x-p_j\|_2 \tag{21.6.2}$$

Since the square root is monotone, the same index is obtained by minimizing the squared distance:

$$j^\ast=\arg\min{1\le j\le N}\|x-p_j\|_2^2\tag{21.6.3}$$

This form is usually preferred in implementation because it avoids unnecessary square-root evaluations.

A closely related query is the $k$-nearest-neighbor problem:

$$\mathcal{N}_k(x) = \{p_{j_1},\ldots,p_{j_k}\}\tag{21.6.4}$$

where,

$$\|x-p_{j_1}\|_2\le\|x-p_{j_2}\|_2\le\cdots\le\|x-p_{j_k}\|_2\tag{21.6.5}$$

and the selected points have the $k$ smallest distances from $x$. This query appears in point-cloud smoothing, local regression, manifold learning, kernel approximation, and direct point-cloud-based numerical analysis.

For particle, meshless, and compact-support kernel methods, fixed-radius search is often more natural than single nearest-neighbor search. Given a radius $r>0$, the radius neighborhood of $x$ is:

$$\mathcal{N}_r(x) = \{p_j\in P:\|x-p_j\|_2\le r\} \tag{21.6.6}$$

Equivalently,

$$\mathcal{N}_r(x) = \{p_j\in P:\|x-p_j\|_2^2\le r^2\}\tag{21.6.7}$$

In smoothed particle hydrodynamics, radial basis function approximation, kernel density estimation, and meshless discretizations, this query directly determines the local stencil or interaction set.

The brute-force method evaluates the distance from $x$ to every point in $P$. For one nearest-neighbor query, this costs:

$$O(Nd)\tag{21.6.8}$$

arithmetic operations, or simply $O(N)$ when the dimension is fixed. For $M$ independent queries, the total cost is:

$$O(MNd) \tag{21.6.9}$$

The brute-force method has two advantages: it is simple and it is exact. It also has predictable memory access when points are stored contiguously. For small $N$, low query counts, or high-dimensional data where tree pruning is ineffective, brute force may be competitive. However, for large-scale point clouds, particle simulations, and repeated interpolation queries, the $O(N)$ cost per query is usually unacceptable.

Spatial indexing attempts to reduce the number of candidate points examined. A spatial index can be viewed as a data structure that maps a query $x$ to a smaller candidate set,

$$C(x)\subseteq P\tag{21.6.10}$$

such that the exact answer is contained in $C(x)$, or such that $C(x)$ provides an acceptable approximate answer. Exact methods require:

$$p_{j^\ast}\in C(x) \tag{21.6.11}$$

whereas approximate methods relax this requirement. For example, an approximate nearest neighbor $\widetilde{p}$ may satisfy:

$$
\|x-\widetilde{p}\|_2
\le
(1+\varepsilon)
\min_{1\le j\le N}
\|x-p_j\|_2
\tag{21.6.12}
$$

for a prescribed approximation parameter $\varepsilon\ge 0$. This distinction between exact and approximate search becomes increasingly important in high dimensions, where exact nearest-neighbor queries may lose their practical advantage because distances become less discriminative (Ting et al., 2024).

### Rust Implementation

Following the discussion in Section 21.6.1 on nearest-neighbor, k-nearest-neighbor, and fixed-radius spatial queries, Program 21.6.1 provides a practical implementation of brute-force spatial search for finite point sets in $\mathbb{R}^d$. The program demonstrates the direct evaluation strategy described by Equations (21.6.2)–(21.6.9), where distances from a query point are computed against every point in the data set without the use of auxiliary spatial indexing structures. Although this approach has linear complexity per query, it remains important as an exact reference implementation and as a competitive method for small data sets, low query counts, or high-dimensional problems where tree-based pruning may lose effectiveness. The implementation emphasizes the use of squared Euclidean distances to avoid unnecessary square-root evaluations, thereby matching the computational formulation introduced in Equations (21.6.3) and (21.6.7). In addition to exact nearest-neighbor search, the program also demonstrates k-nearest-neighbor queries and fixed-radius neighborhood construction, both of which form the computational basis of particle methods, point-cloud processing, kernel approximation, and meshless discretization techniques.

At the core of the implementation is the `Point` structure, which represents a point in $\mathbb{R}^d$ through a dynamically sized vector of floating-point coordinates. This design allows the same search framework to operate in arbitrary spatial dimensions without modifying the underlying algorithms. The associated `dimension` method provides a consistency check to ensure that all distance computations are performed between points of equal dimension. The `Neighbor` structure stores the index of a candidate point together with its squared Euclidean distance from the query point, allowing the program to retain both geometric and indexing information during search operations.

The `squared_distance` function implements the Euclidean distance formulation used throughout Section 21.6.1. Instead of evaluating the full Euclidean norm, the function computes only the squared distance between two points, directly reflecting Equation (21.6.3). Because the square-root operation is monotone, minimizing squared distance yields the same nearest-neighbor ordering as minimizing the true Euclidean distance while avoiding unnecessary computational expense. The function iterates coordinate-wise through the point vectors, accumulates the squared coordinate differences, and returns the total squared norm. This formulation is used consistently throughout the program for nearest-neighbor, k-nearest-neighbor, and radius-search queries.

The `nearest_neighbor` function provides an implementation of the exact nearest-neighbor problem defined by Equations (21.6.2) and (21.6.3). The algorithm sequentially scans all points in the set $P$, computes the squared distance from the query point to each candidate point, and retains the point with the smallest observed distance. Since every point is examined exactly once, the computational complexity follows the brute-force behavior described by Equation (21.6.8). Although asymptotically expensive for very large data sets, this method has the advantage of simplicity, exactness, and predictable memory access patterns because the points are stored contiguously in memory.

The `k_nearest_neighbors` function extends the brute-force approach to the k-nearest-neighbor problem introduced in Equations (21.6.4) and (21.6.5). The function first computes the squared distance from the query point to every point in the data set and stores the resulting candidate neighbors in a vector. The vector is then sorted according to increasing squared distance, after which only the first $k$ entries are retained. This implementation provides an exact ordering of neighbors and demonstrates the geometric interpretation of local neighborhoods used in manifold learning, local regression, point-cloud smoothing, and meshless numerical methods. Although sorting introduces additional computational cost, the implementation remains conceptually straightforward and serves as a useful reference against which more advanced indexed methods can later be compared.

The `radius_search` function implements the fixed-radius neighborhood query defined by Equations (21.6.6) and (21.6.7). Rather than identifying only the closest point, the function returns all points whose squared distance from the query point does not exceed $r^2$. The use of squared distances again avoids square-root evaluations and ensures consistency with the theoretical formulation. The implementation iterates through the point set, filters candidate points according to the radius criterion, and stores all valid neighbors in sorted order. This form of query is especially important in particle methods, compact-support kernels, radial basis function approximation, and smoothed particle hydrodynamics, where the resulting neighborhood directly defines the local interaction stencil.

The auxiliary printing functions, `print_point` and `print_neighbors`, provide formatted diagnostic output for visual verification of the search results. These functions separate presentation logic from the search algorithms themselves, improving program clarity and maintainability. The formatting explicitly reports both squared distances and Euclidean distances, making it possible to verify the correspondence between the theoretical definitions and the computed numerical results.

The `main` function serves to demonstrate the behavior of brute-force spatial search using a small two-dimensional point cloud. It begins by constructing a finite point set $P\subset\mathbb{R}^2$ together with a query point $x$. The program then evaluates three representative spatial queries: exact nearest-neighbor search, exact k-nearest-neighbor search, and exact fixed-radius search. For the nearest-neighbor query, the program identifies the point minimizing the squared Euclidean distance from the query location. For the k-nearest-neighbor query, the program sorts all candidate points according to distance and extracts the closest $k$ neighbors. Finally, for the radius-search query, the program identifies all points satisfying the radius condition of Equation (21.6.7). The printed output illustrates how local neighborhoods emerge from direct geometric distance calculations and provides a baseline reference against which more sophisticated spatial indexing methods such as uniform grids, k-d trees, octrees, and BVHs can later be evaluated.

```rust
// Program 21.6.1: Brute-Force Nearest-Neighbor, k-Nearest-Neighbor,
// and Fixed-Radius Search
//
// Problem statement:
// Given a finite point set P = {p_1, p_2, ..., p_N} in R^d and a query
// point x, compute the exact nearest neighbor, the k nearest neighbors,
// and the fixed-radius neighborhood using squared Euclidean distances.
// The implementation follows equations (21.6.2) to (21.6.7), avoiding
// square roots whenever possible.

#[derive(Clone, Debug)]
struct Point {
    coords: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Neighbor {
    index: usize,
    squared_distance: f64,
}

impl Point {
    fn new(coords: Vec<f64>) -> Self {
        Self { coords }
    }

    fn dimension(&self) -> usize {
        self.coords.len()
    }
}

fn squared_distance(a: &Point, b: &Point) -> f64 {
    assert_eq!(
        a.dimension(),
        b.dimension(),
        "points must have the same dimension"
    );

    a.coords
        .iter()
        .zip(b.coords.iter())
        .map(|(ai, bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum()
}

fn nearest_neighbor(points: &[Point], query: &Point) -> Option<Neighbor> {
    points
        .iter()
        .enumerate()
        .map(|(index, p)| Neighbor {
            index,
            squared_distance: squared_distance(p, query),
        })
        .min_by(|a, b| {
            a.squared_distance
                .partial_cmp(&b.squared_distance)
                .unwrap()
                .then_with(|| a.index.cmp(&b.index))
        })
}

fn k_nearest_neighbors(points: &[Point], query: &Point, k: usize) -> Vec<Neighbor> {
    let mut neighbors: Vec<Neighbor> = points
        .iter()
        .enumerate()
        .map(|(index, p)| Neighbor {
            index,
            squared_distance: squared_distance(p, query),
        })
        .collect();

    neighbors.sort_by(|a, b| {
        a.squared_distance
            .partial_cmp(&b.squared_distance)
            .unwrap()
            .then_with(|| a.index.cmp(&b.index))
    });

    neighbors.truncate(k.min(neighbors.len()));
    neighbors
}

fn radius_search(points: &[Point], query: &Point, radius: f64) -> Vec<Neighbor> {
    assert!(radius >= 0.0, "radius must be nonnegative");

    let radius_squared = radius * radius;

    let mut neighbors: Vec<Neighbor> = points
        .iter()
        .enumerate()
        .filter_map(|(index, p)| {
            let dist2 = squared_distance(p, query);

            if dist2 <= radius_squared {
                Some(Neighbor {
                    index,
                    squared_distance: dist2,
                })
            } else {
                None
            }
        })
        .collect();

    neighbors.sort_by(|a, b| {
        a.squared_distance
            .partial_cmp(&b.squared_distance)
            .unwrap()
            .then_with(|| a.index.cmp(&b.index))
    });

    neighbors
}

fn print_point(label: &str, p: &Point) {
    print!("{label} = (");
    for (i, value) in p.coords.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{value:.6}");
    }
    println!(")");
}

fn print_neighbors(title: &str, points: &[Point], neighbors: &[Neighbor]) {
    println!("{title}");
    println!("{}", "-".repeat(title.len()));

    for neighbor in neighbors {
        print!("index {:>2}, ", neighbor.index);
        print!("squared distance = {:>12.6}, ", neighbor.squared_distance);
        print_point("point", &points[neighbor.index]);
    }

    if neighbors.is_empty() {
        println!("no points found");
    }

    println!();
}

fn main() {
    let points = vec![
        Point::new(vec![0.0, 0.0]),
        Point::new(vec![1.0, 2.0]),
        Point::new(vec![2.0, 1.0]),
        Point::new(vec![3.0, 3.0]),
        Point::new(vec![4.0, 2.0]),
        Point::new(vec![5.0, 4.0]),
        Point::new(vec![2.5, 2.5]),
    ];

    let query = Point::new(vec![2.2, 2.0]);
    let k = 3;
    let radius = 1.5;

    println!("Brute-Force Spatial Search");
    println!("==========================\n");

    println!("Point Set");
    println!("---------");
    for (i, p) in points.iter().enumerate() {
        print!("p_{i:<2} ");
        print_point("", p);
    }
    println!();

    println!("Query Point");
    println!("-----------");
    print_point("x", &query);
    println!();

    if let Some(nn) = nearest_neighbor(&points, &query) {
        println!("Nearest Neighbor");
        println!("----------------");
        println!("index              = {}", nn.index);
        println!("squared distance   = {:.6}", nn.squared_distance);
        println!("distance           = {:.6}", nn.squared_distance.sqrt());
        print_point("nearest point", &points[nn.index]);
        println!();
    }

    let knn = k_nearest_neighbors(&points, &query, k);
    print_neighbors(
        &format!("{k}-Nearest Neighbors"),
        &points,
        &knn,
    );

    let radius_neighbors = radius_search(&points, &query, radius);
    println!("Fixed-Radius Search");
    println!("-------------------");
    println!("radius             = {:.6}", radius);
    println!("radius squared     = {:.6}", radius * radius);
    println!();

    print_neighbors(
        "Points Inside Radius",
        &points,
        &radius_neighbors,
    );
}
```

Program 21.6.1 demonstrates the fundamental computational structure underlying exact spatial search by implementing brute-force nearest-neighbor, k-nearest-neighbor, and fixed-radius queries directly from their mathematical definitions. The implementation reflects the central ideas developed in Section 21.6.1: spatial queries can be formulated entirely in terms of geometric distance evaluation, and squared Euclidean distances provide a computationally efficient representation without altering the ordering of candidate points.

The examples illustrate three distinct neighborhood-search paradigms that recur throughout scientific computing. Exact nearest-neighbor search identifies the single closest geometric entity to a query point, k-nearest-neighbor search constructs ordered local neighborhoods for interpolation and approximation, and fixed-radius search generates interaction sets determined by physical support radii or smoothing lengths. Together, these queries form the basis of many algorithms in particle simulation, point-cloud processing, kernel methods, collision detection, and meshless numerical discretization.

Although the brute-force approach scales linearly with the number of stored points, it remains valuable as an exact baseline implementation against which accelerated spatial indexing methods can be validated. The program also illustrates the broader computational principle emphasized throughout this section: spatial search consists fundamentally of candidate generation followed by exact geometric verification. Subsequent implementations based on uniform grids, voxel hashing, k-d trees, octrees, and bounding-volume hierarchies will retain this same conceptual structure while reducing the number of candidate points that must be examined during each query.

## 21.6.2. Uniform Grids, Voxel Hashing, and Locality-Aware Layouts

Uniform grids are the simplest spatial index for fixed-radius and local-neighborhood queries. Suppose the computational domain is divided into axis-aligned cells of side length $h>0$. For a point $p\in\mathbb{R}^d$, its grid index is:

$$
g(p)
=
\left(
\left\lfloor \frac{p_1-x_{0,1}}{h}\right\rfloor,
\ldots,
\left\lfloor \frac{p_d-x_{0,d}}{h}\right\rfloor
\right)
\in \mathbb{Z}^d
\tag{21.6.13}
$$

where $x_0$ is the grid origin. Points are stored in buckets according to their grid index. A fixed-radius query then inspects only the bucket containing $x$ and nearby buckets whose cells may intersect the ball $B(x,r)$.

If the grid spacing satisfies $h\approx r$, then a radius search in $d$ dimensions checks a fixed number of neighboring buckets. For example, if $h=r$, then all points within distance $r$ of $x$ must lie in the grid cell of $x$ or in one of its adjacent cells. Thus, the number of buckets checked is bounded by $3^d$ for the immediate-neighbor stencil. More generally, if $m=\lceil r/h\rceil$, the bucket stencil has at most $(2m+1)^d$ cells. The cost of a query is therefore approximately proportional to the number of candidate points in these buckets $O(|C(x)|)$, after constant-time or near-constant-time bucket lookup.

Uniform grids are effective when the point distribution is reasonably uniform and the query radius is known in advance. They can perform poorly when the data are highly clustered, when density varies by orders of magnitude, or when many different query radii are required. In such cases, some buckets may contain too many points while others remain empty. Sparse grids and voxel hashing address this by storing only occupied cells. A hash key can be built from the integer grid coordinates:

$$k=g(p)\in\mathbb{Z}^d,\tag{21.6.14}$$

and a hash map stores the list of points associated with each occupied key. This avoids allocating a full dense grid over a large empty domain.

For large point clouds, memory locality is often as important as the number of distance calculations. If points that are close in space are also close in memory, then cache behavior improves. Space-filling curves provide one way to convert geometric locality into memory locality. In three dimensions, a Morton index interleaves the bits of integer grid coordinates. If $g(p)=(i,j,k)$, then the Morton code $M(i,j,k)$ is obtained by interleaving the binary digits of $i$, $j$, and $k$. Sorting points by this code produces an order in which nearby grid cells tend to be stored near each other in memory. Hilbert curves provide a more locality-preserving ordering, although they are more complex to compute.

Linear octrees use a similar idea. Instead of storing a pointer-rich recursive tree, cells are encoded by integer keys and stored in sorted arrays. A query can then traverse ranges of keys associated with spatially nearby cells. Recent work on space-filling curves, linear octrees, and high-performance point indexing demonstrates that such layouts can reduce cache misses and improve neighborhood search performance for large three-dimensional point clouds and LiDAR-scale data (Viñambres et al., 2026; Laso and Yermo, 2026).

The scientific-computing lesson is that a grid is not merely a simple approximation to a tree. For fixed-radius interactions, particle methods, local stencil construction, and point-cloud neighborhoods, grid-like structures are often the best match to the query. Their strength is not an $O(\log N)$ theoretical bound, but the combination of direct indexing, local scanning, contiguous storage, and predictable memory access.

### Rust Implementation

Following the discussion in Section 21.6.2 on uniform grids, voxel hashing, and locality-aware spatial layouts, Program 21.6.2 provides a practical implementation of sparse grid-based fixed-radius search using a voxel hash structure. The program demonstrates how the grid indexing relation introduced in Equation (21.6.13) can be used to organize a point cloud into spatial buckets, thereby reducing the number of candidate points examined during neighborhood queries. Instead of scanning every point in the data set as in the brute-force method of Section 21.6.1, the implementation stores only occupied cells in a hash map according to the voxel-hashing formulation of Equation (21.6.14). Radius queries are then performed by inspecting only nearby grid cells whose buckets may intersect the query ball. The implementation combines coarse spatial pruning with exact squared-distance verification, illustrating the central computational principle underlying efficient neighborhood search in particle methods, meshless discretizations, point-cloud processing, and local interpolation algorithms. The program also emphasizes locality-aware storage and candidate filtering, both of which are essential for high-performance scientific computing on large geometric data sets.

At the core of the implementation is the `Point` structure, which stores coordinates in a dynamically sized vector and therefore supports spatial search in arbitrary dimensions. The `Neighbor` structure stores both the index of a candidate point and its squared distance from the query location, allowing the program to retain exact geometric information during search operations. The `CellIndex` structure represents the integer grid coordinates associated with a voxel cell and serves as the key type used in the voxel hash table. Because the structure derives the `Hash`, `Eq`, and `PartialEq` traits, it can be used directly as a key in Rust’s `HashMap` container for sparse spatial indexing.

The principal data structure of the implementation is the `VoxelHashGrid`, which stores the point cloud, the grid origin $x_0$, the grid spacing $h$, and the sparse hash map containing occupied buckets. During construction, the `build` function assigns each point to its corresponding voxel cell according to the grid-index relation of Equation (21.6.13). Rather than allocating a dense multidimensional array over the entire computational domain, the implementation stores only occupied cells in the hash map, directly reflecting the sparse voxel-hashing strategy described by Equation (21.6.14). This approach avoids unnecessary memory allocation in large sparse domains and provides near-constant-time bucket lookup.

The `cell_index` function implements the spatial discretization formula introduced in Equation (21.6.13). For each coordinate direction, the function subtracts the grid origin, divides by the cell spacing, and applies the floor operation to obtain an integer voxel coordinate. The resulting integer tuple identifies the bucket to which the point belongs. Because the implementation uses signed integer indices, the method naturally supports points located in regions with negative coordinates, provided that the grid origin is defined consistently.

The `radius_search` function provides the central neighborhood-search algorithm of the program. Given a query point $x$ and a radius $r$, the function first determines the voxel cell containing the query point and computes the stencil radius:

$$m=\left\lceil \frac{r}{h}\right\rceil,$$

which determines the neighboring buckets that must be inspected. The function then enumerates all voxel cells within the $(2m+1)^d$ stencil surrounding the query cell, matching the neighborhood-search strategy described in Section 21.6.2. Candidate points from these occupied buckets are examined individually using exact squared-distance checks corresponding to Equation (21.6.7). Only those points satisfying the radius criterion are retained in the final neighbor list. This two-stage procedure demonstrates the fundamental decomposition of spatial search into candidate generation followed by exact local verification.

The `enumerate_neighbor_cells` function recursively generates all voxel cells contained within the local stencil surrounding the query cell. The recursion proceeds dimension by dimension, constructing all combinations of integer offsets in the range $[-m,m]$. This implementation provides a general multidimensional stencil generator and directly reflects the geometric neighborhood structure associated with uniform-grid search. Because the number of visited cells depends only on the stencil radius and the spatial dimension, the candidate-generation cost becomes largely independent of the total number of stored points when the point distribution is reasonably uniform.

The `squared_distance` function computes the squared Euclidean distance between two points and is used consistently throughout the implementation to avoid unnecessary square-root evaluations. This follows the computational strategy introduced previously in Equation (21.6.7), where comparisons are performed entirely using squared distances. The implementation iterates coordinate-wise through the point vectors, accumulates the squared coordinate differences, and returns the resulting squared norm. This formulation improves efficiency while preserving the exact ordering and acceptance conditions required for radius search.

The auxiliary function `brute_force_radius_search` provides a direct verification mechanism against the voxel-hash search results. By evaluating the radius condition against every point in the data set, it produces the exact brute-force neighborhood described in Section 21.6.1. The comparison function `same_neighbor_indices` then verifies that the voxel-hash search produces the same accepted neighbor set as the brute-force reference implementation. This verification step demonstrates that the voxel hash acts only as a candidate-generation mechanism and does not alter the exact geometric acceptance criterion.

The diagnostic printing functions separate presentation logic from the spatial-search algorithms themselves. Functions such as `print_point`, `print_cell`, and `print_neighbors` provide structured output for points, voxel coordinates, and neighborhood results, while `print_bucket_summary` reports the occupied voxel cells and their associated point indices. These diagnostics illustrate the sparse nature of the voxel hash and make it possible to visualize how points are distributed among buckets. The query diagnostics additionally report the number of inspected cells, candidate points examined, and accepted neighbors, thereby exposing the reduction in search cost achieved through spatial indexing.

The `main` function serves to demonstrate sparse uniform-grid construction and voxel-hash radius search on a two-dimensional point cloud. It begins by constructing a finite set of sample points together with a grid origin and cell spacing. The point cloud is inserted into the voxel hash structure, after which the occupied buckets and their associated point indices are printed for inspection. The program then performs an exact fixed-radius query centered at a prescribed query point. The resulting neighborhood is computed first through voxel-hash candidate generation and then verified against a brute-force radius search over the entire point set. The diagnostic output illustrates how only a small subset of candidate points must be examined, even though the exact neighborhood produced is identical to the brute-force result. This demonstrates the essential computational advantage of grid-based spatial indexing for fixed-radius interaction queries.

```rust
// Program 21.6.2: Sparse Uniform Grid and Voxel Hashing for Exact Fixed-Radius
// Spatial Search
//
// Problem statement:
// Given a finite point set P in R^d, construct a sparse uniform grid by
// assigning each point to an integer grid cell according to
//
//     g(p) = floor((p - x_0) / h),
//
// as in Equation (21.6.13). Store only occupied cells in a hash map, as in
// Equation (21.6.14). For a fixed-radius query, inspect only the grid cell
// containing the query point and the neighboring cells that can intersect the
// query ball. Candidate points are then filtered by an exact squared-distance
// check, following the radius-search condition in Equation (21.6.7).

use std::collections::HashMap;

#[derive(Clone, Debug)]
struct Point {
    coords: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Neighbor {
    index: usize,
    squared_distance: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CellIndex {
    coords: Vec<i64>,
}

#[derive(Debug)]
struct VoxelHashGrid {
    points: Vec<Point>,
    origin: Point,
    cell_size: f64,
    buckets: HashMap<CellIndex, Vec<usize>>,
}

impl Point {
    fn new(coords: Vec<f64>) -> Self {
        Self { coords }
    }

    fn dimension(&self) -> usize {
        self.coords.len()
    }
}

impl CellIndex {
    fn new(coords: Vec<i64>) -> Self {
        Self { coords }
    }

    fn dimension(&self) -> usize {
        self.coords.len()
    }
}

impl VoxelHashGrid {
    fn new(points: Vec<Point>, origin: Point, cell_size: f64) -> Self {
        assert!(cell_size > 0.0, "cell size must be positive");

        if let Some(first) = points.first() {
            assert_eq!(
                first.dimension(),
                origin.dimension(),
                "point dimension must match grid-origin dimension"
            );

            for p in &points {
                assert_eq!(
                    p.dimension(),
                    first.dimension(),
                    "all points must have the same dimension"
                );
            }
        }

        let mut grid = Self {
            points,
            origin,
            cell_size,
            buckets: HashMap::new(),
        };

        grid.build();
        grid
    }

    fn build(&mut self) {
        self.buckets.clear();

        for index in 0..self.points.len() {
            let cell = self.cell_index(&self.points[index]);
            self.buckets.entry(cell).or_default().push(index);
        }
    }

    fn cell_index(&self, point: &Point) -> CellIndex {
        assert_eq!(
            point.dimension(),
            self.origin.dimension(),
            "point dimension must match grid-origin dimension"
        );

        let coords = point
            .coords
            .iter()
            .zip(self.origin.coords.iter())
            .map(|(x, x0)| ((x - x0) / self.cell_size).floor() as i64)
            .collect();

        CellIndex::new(coords)
    }

    fn radius_search(&self, query: &Point, radius: f64) -> SearchReport {
        assert!(radius >= 0.0, "radius must be nonnegative");
        assert_eq!(
            query.dimension(),
            self.origin.dimension(),
            "query dimension must match grid-origin dimension"
        );

        let query_cell = self.cell_index(query);
        let stencil_radius = (radius / self.cell_size).ceil() as i64;
        let radius_squared = radius * radius;

        let mut neighbor_cells = Vec::new();
        enumerate_neighbor_cells(
            &query_cell,
            stencil_radius,
            0,
            &mut Vec::new(),
            &mut neighbor_cells,
        );

        let mut candidates_examined = 0usize;
        let mut neighbors = Vec::new();

        for cell in &neighbor_cells {
            if let Some(indices) = self.buckets.get(cell) {
                for &point_index in indices {
                    candidates_examined += 1;

                    let dist2 = squared_distance(&self.points[point_index], query);

                    if dist2 <= radius_squared {
                        neighbors.push(Neighbor {
                            index: point_index,
                            squared_distance: dist2,
                        });
                    }
                }
            }
        }

        neighbors.sort_by(|a, b| {
            a.squared_distance
                .partial_cmp(&b.squared_distance)
                .unwrap()
                .then_with(|| a.index.cmp(&b.index))
        });

        SearchReport {
            query_cell,
            stencil_radius,
            cells_checked: neighbor_cells.len(),
            candidates_examined,
            neighbors,
        }
    }

    fn occupied_cell_count(&self) -> usize {
        self.buckets.len()
    }

    fn largest_bucket_size(&self) -> usize {
        self.buckets
            .values()
            .map(|bucket| bucket.len())
            .max()
            .unwrap_or(0)
    }

    fn print_bucket_summary(&self) {
        println!("Voxel Hash Grid Summary");
        println!("-----------------------");
        println!("number of points        = {}", self.points.len());
        println!("dimension               = {}", self.origin.dimension());
        println!("cell size h             = {:.6}", self.cell_size);
        println!("occupied cells          = {}", self.occupied_cell_count());
        println!("largest bucket size     = {}", self.largest_bucket_size());
        println!();

        let mut cells: Vec<_> = self.buckets.iter().collect();
        cells.sort_by(|(a, _), (b, _)| a.coords.cmp(&b.coords));

        println!("Occupied Buckets");
        println!("----------------");
        for (cell, indices) in cells {
            print!("cell ");
            print_cell(cell);
            print!(" -> point indices ");
            print_index_list(indices);
            println!();
        }
        println!();
    }
}

#[derive(Debug)]
struct SearchReport {
    query_cell: CellIndex,
    stencil_radius: i64,
    cells_checked: usize,
    candidates_examined: usize,
    neighbors: Vec<Neighbor>,
}

fn squared_distance(a: &Point, b: &Point) -> f64 {
    assert_eq!(
        a.dimension(),
        b.dimension(),
        "points must have the same dimension"
    );

    a.coords
        .iter()
        .zip(b.coords.iter())
        .map(|(ai, bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum()
}

fn enumerate_neighbor_cells(
    center: &CellIndex,
    stencil_radius: i64,
    axis: usize,
    current_offsets: &mut Vec<i64>,
    cells: &mut Vec<CellIndex>,
) {
    assert!(stencil_radius >= 0, "stencil radius must be nonnegative");

    if axis == center.dimension() {
        let coords = center
            .coords
            .iter()
            .zip(current_offsets.iter())
            .map(|(c, offset)| c + offset)
            .collect();

        cells.push(CellIndex::new(coords));
        return;
    }

    for offset in -stencil_radius..=stencil_radius {
        current_offsets.push(offset);
        enumerate_neighbor_cells(center, stencil_radius, axis + 1, current_offsets, cells);
        current_offsets.pop();
    }
}

fn brute_force_radius_search(points: &[Point], query: &Point, radius: f64) -> Vec<Neighbor> {
    let radius_squared = radius * radius;

    let mut neighbors: Vec<Neighbor> = points
        .iter()
        .enumerate()
        .filter_map(|(index, p)| {
            let dist2 = squared_distance(p, query);

            if dist2 <= radius_squared {
                Some(Neighbor {
                    index,
                    squared_distance: dist2,
                })
            } else {
                None
            }
        })
        .collect();

    neighbors.sort_by(|a, b| {
        a.squared_distance
            .partial_cmp(&b.squared_distance)
            .unwrap()
            .then_with(|| a.index.cmp(&b.index))
    });

    neighbors
}

fn same_neighbor_indices(a: &[Neighbor], b: &[Neighbor]) -> bool {
    let a_indices: Vec<usize> = a.iter().map(|n| n.index).collect();
    let b_indices: Vec<usize> = b.iter().map(|n| n.index).collect();

    a_indices == b_indices
}

fn print_point(label: &str, point: &Point) {
    print!("{label}(");
    for (i, value) in point.coords.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{value:.6}");
    }
    println!(")");
}

fn print_cell(cell: &CellIndex) {
    print!("(");
    for (i, value) in cell.coords.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{value}");
    }
    print!(")");
}

fn print_index_list(indices: &[usize]) {
    print!("[");
    for (i, index) in indices.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{index}");
    }
    print!("]");
}

fn print_neighbors(title: &str, points: &[Point], neighbors: &[Neighbor]) {
    println!("{title}");
    println!("{}", "-".repeat(title.len()));

    if neighbors.is_empty() {
        println!("no neighbors found");
        println!();
        return;
    }

    for neighbor in neighbors {
        print!("index {:>2}, ", neighbor.index);
        print!("squared distance = {:>10.6}, ", neighbor.squared_distance);
        print_point("point = ", &points[neighbor.index]);
    }

    println!();
}

fn main() {
    let points = vec![
        Point::new(vec![0.0, 0.0]),
        Point::new(vec![1.0, 2.0]),
        Point::new(vec![2.0, 1.0]),
        Point::new(vec![3.0, 3.0]),
        Point::new(vec![4.0, 2.0]),
        Point::new(vec![5.0, 4.0]),
        Point::new(vec![2.5, 2.5]),
        Point::new(vec![7.0, 7.0]),
        Point::new(vec![8.0, 7.5]),
        Point::new(vec![-1.0, -0.5]),
    ];

    let origin = Point::new(vec![0.0, 0.0]);
    let cell_size = 1.5;
    let query = Point::new(vec![2.2, 2.0]);
    let radius = 1.5;

    println!("Sparse Uniform Grid and Voxel Hashing");
    println!("======================================\n");

    println!("Point Set");
    println!("---------");
    for (i, p) in points.iter().enumerate() {
        print!("p_{:<2} = ", i);
        print_point("", p);
    }
    println!();

    println!("Grid Parameters");
    println!("---------------");
    println!("cell size h          = {:.6}", cell_size);
    print_point("origin x_0 = ", &origin);
    println!();

    let grid = VoxelHashGrid::new(points, origin, cell_size);
    grid.print_bucket_summary();

    println!("Fixed-Radius Query");
    println!("------------------");
    print_point("query x = ", &query);
    println!("radius r            = {:.6}", radius);
    println!("radius squared      = {:.6}", radius * radius);
    println!();

    let report = grid.radius_search(&query, radius);

    println!("Voxel Hash Query Diagnostics");
    println!("----------------------------");
    print!("query cell g(x)     = ");
    print_cell(&report.query_cell);
    println!();
    println!("stencil radius m    = {}", report.stencil_radius);
    println!("cells checked       = {}", report.cells_checked);
    println!("candidates examined = {}", report.candidates_examined);
    println!(
        "accepted neighbors  = {}",
        report.neighbors.len()
    );
    println!();

    print_neighbors(
        "Voxel-Hash Radius Search Result",
        &grid.points,
        &report.neighbors,
    );

    let brute_force_neighbors = brute_force_radius_search(&grid.points, &query, radius);

    print_neighbors(
        "Brute-Force Verification Result",
        &grid.points,
        &brute_force_neighbors,
    );

    println!("Verification");
    println!("------------");
    println!(
        "voxel hash agrees with brute force = {}",
        same_neighbor_indices(&report.neighbors, &brute_force_neighbors)
    );
}
```

Program 21.6.2 demonstrates how sparse uniform grids and voxel hashing transform spatial search from a global scanning problem into a localized neighborhood-query problem. By assigning points to voxel buckets according to integer grid coordinates, the implementation reduces the number of candidate points examined during a radius query while preserving exact geometric correctness through final squared-distance verification.

The example illustrates the central advantage of grid-based spatial indexing for fixed-radius interactions. Instead of evaluating distances against every point in the data set, the search algorithm inspects only those voxel cells whose spatial extent may intersect the query ball. The resulting reduction in candidate count reflects the principle discussed in Section 21.6.2: when the grid spacing is comparable to the interaction radius, the search complexity becomes approximately proportional to the number of local candidate points rather than the total size of the point cloud.

The implementation also demonstrates the practical importance of sparse storage and locality-aware layouts. By storing only occupied cells in a hash map, the voxel hash avoids allocating large empty grids while still providing efficient bucket lookup. At the same time, the separation between coarse candidate generation and exact local filtering preserves numerical correctness and provides a clean foundation for more advanced spatial-search structures such as Morton-order grids, linear octrees, k-d trees, and bounding-volume hierarchies. The program therefore serves both as a practical neighborhood-search implementation and as an introduction to locality-aware geometric indexing techniques used throughout modern scientific computing.

## 21.6.3. Tree-Based Search: (k)-d Trees, Ball Trees, Octrees, and BVHs

Tree-based spatial indexes recursively subdivide either the point set or the ambient space. The $k$-d tree is the classical example for point data in $\mathbb{R}^d$. Each internal node stores a splitting coordinate $s\in{1,\ldots,d}$ and a splitting value $\alpha$, dividing the point set into:

$$
P_{\mathrm{left}}
=
\{p\in P : p_s \le \alpha\},
\\
P_{\mathrm{right}}
=
\{p\in P : p_s > \alpha\}
\tag{21.6.15}
$$

The splitting coordinate may be chosen cyclically, or it may be selected as the coordinate with largest spread:

$$
s
=
\arg\max_{1\le m\le d}
\left(
\max_{p\in P} p_m
-
\min_{p\in P} p_m
\right)
\tag{21.6.16}
$$

A balanced $k$-d tree can be built in approximately $O(N\log N)$ time, or in $O(N\log^2N)$ depending on the sorting strategy. In low dimensions and for well-distributed data, nearest-neighbor queries often take approximately $O(\log N)$ time on average, although this is not a uniform guarantee independent of dimension and distribution.

Nearest-neighbor search in a $k$-d tree uses pruning. During traversal, the algorithm keeps the best squared distance found so far,

$$
\rho^2
=
\min_{p\in C_{\mathrm{visited}}}
\|x-p\|_2^2
\tag{21.6.17}
$$

A subtree whose bounding box $B$ satisfies,

$$\operatorname{dist}(x,B)^2>\rho^2\tag{21.6.18}$$

cannot contain a closer point and may be skipped. For an axis-aligned bounding box,

$$B=[\ell_1,u_1]\times\cdots\times[\ell_d,u_d],\tag{21.6.19}$$

the squared distance from $x$ to $B$ is:

$$
\operatorname{dist}(x,B)^2
=
\sum_{m=1}^d \delta_m^2
\tag{21.6.20}
$$

where,

$$
\delta_m
=
\begin{cases}
\ell_m - x_m, & x_m < \ell_m, \\
0, & \ell_m \le x_m \le u_m, \\
x_m - u_m, & x_m > u_m
\end{cases}
\tag{21.6.21}
$$

This formula is fundamental not only for $k$-d trees, but also for octrees and bounding volume hierarchies.

Ball trees use hyperspheres rather than axis-aligned boxes. A node stores a center $c$ and radius $R$ such that all points in the node lie in:

$$
B(c,R)
=
\left\{
y\in\mathbb{R}^d :
\|y-c\|_2 \le R
\right\}
\tag{21.6.22}
$$

A nearest-neighbor query can prune the node if:

$$\|x-c\|_2-R>\rho \tag{21.6.23}$$

Ball trees may be useful when clusters are more naturally spherical than axis-aligned, although their efficiency remains data-dependent.

Octrees and quadtrees subdivide space rather than splitting by point medians. In two dimensions, a quadtree divides an axis-aligned square into four children; in three dimensions, an octree divides a cube into eight children. If a root cube has side length $L$, then after level $\ell$ subdivision, the side length is:

$$
h_\ell
=
\frac{L}{2^\ell}
\tag{21.6.24}
$$

This makes octrees natural for multiresolution spatial organization, adaptive sampling, level-of-detail representations, and point-cloud neighborhoods. However, pointer-based octrees can suffer from poor memory locality. Linear octrees improve this by encoding cells through Morton keys and storing them in contiguous arrays.

Bounding volume hierarchies, or BVHs, organize geometric objects by enclosing them in simple bounding volumes, usually axis-aligned bounding boxes. For a set of primitives $\mathcal{G}={G_1,\ldots,G_M}$, a node bounding box satisfies:

$$
G_i \subseteq B
\quad
\text{for all primitives } G_i \text{ assigned to the node}
\tag{21.6.25}
$$

Intersection, containment, and nearest-distance queries prune nodes whose bounding boxes cannot contribute. BVHs are widely associated with graphics, but the same bounding-and-pruning principle is important in scientific computing for mesh intersection, ray tracing through simulation domains, collision/contact detection, and geometric search over cells. Modern high-performance BVH construction work shows that hierarchy quality, parallel construction, and memory layout are major determinants of practical performance (Benthin et al., 2024).

The important comparison is not that one tree is universally best. A $k$-d tree is natural for low-dimensional point data and exact nearest-neighbor search. A grid or voxel hash is often superior for fixed-radius particle neighborhoods. An octree is useful for adaptive spatial subdivision and multiresolution point clouds. A BVH is natural when the stored objects are not just points, but segments, triangles, boxes, cells, or other geometric primitives. The data structure should be chosen according to the query, dimension, distribution, and downstream numerical method.

### Rust Implementation

Following the discussion in Section 21.6.3 on tree-based spatial indexing, Program 21.6.3 provides a practical implementation of balanced k-d tree nearest-neighbor search with axis-aligned bounding-box pruning. The program demonstrates how recursive spatial subdivision can reduce the number of candidate points examined during nearest-neighbor queries by organizing the point set into a hierarchical structure. In contrast to the brute-force and voxel-hash approaches of Sections 21.6.1 and 21.6.2, the k-d tree recursively partitions the point cloud according to coordinate-aligned splitting planes, thereby enabling geometric pruning during traversal. The implementation follows the largest-spread splitting strategy introduced in Equation (21.6.16) and uses bounding-box distance tests based on Equations (21.6.18)–(21.6.21) to safely eliminate subtrees that cannot contain a closer point than the current best candidate. The program therefore illustrates the central computational principle underlying many hierarchical geometric data structures: recursive subdivision combined with conservative spatial pruning. Although the implementation focuses specifically on k-d trees, the same bounding-and-pruning philosophy forms the basis of octrees, ball trees, and bounding volume hierarchies used throughout scientific computing, point-cloud processing, collision detection, and geometric search.

At the core of the implementation is the `Point` structure, which stores coordinates in a dynamically sized vector and therefore supports spatial search in arbitrary dimensions. The `Neighbor` structure stores the index of a candidate point together with its squared Euclidean distance from the query point, allowing the nearest-neighbor search to retain both geometric and indexing information during traversal. The `Aabb` structure represents an axis-aligned bounding box of the form introduced in Equation (21.6.19), storing lower and upper coordinate bounds for each spatial direction. This structure plays a central role in subtree pruning because it provides a conservative geometric enclosure for all points contained within a node.

The `KdNode` structure represents a node of the balanced k-d tree. Each node stores its bounding box, the index of the median point associated with the node, the splitting coordinate direction, and optional left and right child nodes. This recursive organization directly reflects the partitioning strategy introduced in Equation (21.6.15), where the point set is divided into left and right subsets according to a coordinate-aligned splitting plane. The `KdTree` structure stores the complete point set together with the root node and the spatial dimension, thereby separating the geometric data from the recursive search hierarchy.

The `Aabb::from_points` function constructs a bounding box enclosing all points assigned to a node. For each coordinate direction, the function computes the minimum and maximum coordinate values across the associated point subset and stores the resulting interval bounds. The resulting axis-aligned box conservatively encloses all points belonging to the subtree and therefore enables safe pruning during nearest-neighbor traversal. This bounding-box representation forms the geometric basis of the pruning condition described by Equation (21.6.18).

The `Aabb::squared_distance_to_point` function implements the bounding-box distance formula introduced in Equations (21.6.20) and (21.6.21). For each coordinate direction, the function computes the distance contribution (\\delta_m) between the query coordinate and the nearest point of the bounding interval. If the query coordinate lies inside the interval, the contribution is zero; otherwise, the contribution is the distance to the nearest boundary. The squared contributions are accumulated across all coordinate directions to produce the squared distance between the query point and the bounding box. This quantity is fundamental for safe pruning because any subtree whose bounding-box distance exceeds the current best squared distance cannot contain a closer point.

The `KdTree::new` function constructs the balanced k-d tree from the input point set. After validating dimensional consistency, the function initializes an index array representing the point ordering and recursively builds the hierarchy through the `build_recursive` function. This construction strategy separates the geometric point storage from the tree topology and avoids duplicating point coordinates inside the tree nodes themselves.

The `build_recursive` function implements recursive k-d tree construction. At each level, the function first constructs a bounding box for the current point subset and then selects a splitting coordinate according to the largest-spread strategy described in Equation (21.6.16). The point indices are sorted along this coordinate direction, and the median point is selected as the node representative. The remaining points are partitioned into left and right subsets, which are then processed recursively to construct the child subtrees. This recursive median splitting produces a reasonably balanced hierarchy and reduces the expected search depth for nearest-neighbor queries.

The `largest_spread_axis` function determines the coordinate direction along which the current point subset has the largest geometric extent. For each axis, the function computes the difference between the maximum and minimum coordinate values and selects the axis with the largest spread. This strategy attempts to reduce geometric anisotropy within the resulting subtrees and typically improves pruning effectiveness compared to purely cyclic coordinate selection.

The `squared_distance` function computes the squared Euclidean distance between two points and is used throughout the implementation for nearest-neighbor comparisons. By operating entirely with squared distances, the implementation avoids unnecessary square-root evaluations while preserving the exact ordering of candidate distances, consistent with the nearest-neighbor formulation introduced earlier in Equation (21.6.17).

The `nearest_neighbor` and `search_recursive` functions implement the central nearest-neighbor traversal algorithm. The search maintains the current best squared distance

$$\rho^2 = \min_{p\in C_{\mathrm{visited}}}\|x-p\|_2^2,$$

as introduced in Equation (21.6.17). During traversal, the bounding-box distance of a subtree is first evaluated. If this distance exceeds the current best squared distance, the subtree is safely pruned according to Equation (21.6.18), because it cannot contain a closer point. Otherwise, the node point itself is tested, and the search recursively traverses the child nodes. The implementation visits the subtree on the same side of the splitting plane as the query point first, increasing the likelihood that a strong nearest-neighbor candidate is found early and thereby improving pruning effectiveness for later subtree visits.

The `brute_force_nearest_neighbor` function provides an exact reference implementation against which the k-d tree result can be verified. By scanning every point in the data set and computing the squared distance directly, it reproduces the brute-force nearest-neighbor formulation introduced in Section 21.6.1. The agreement between the brute-force and k-d tree results confirms that the pruning strategy does not alter geometric correctness.

The diagnostic printing functions separate presentation logic from the geometric search algorithms themselves. Functions such as `print_point` and `print_neighbor` provide structured output for points and nearest-neighbor results, while the search statistics stored in the `SearchStats` structure expose the internal behavior of the traversal algorithm. The reported numbers of visited nodes, pruned nodes, and point tests illustrate the computational savings achieved through hierarchical pruning compared to exhaustive brute-force search.

The `main` function serves to demonstrate balanced k-d tree construction and nearest-neighbor search on a two-dimensional point cloud. It begins by constructing a finite set of sample points together with a query point. The point cloud is then organized into a balanced k-d tree using recursive median splitting and largest-spread axis selection. A nearest-neighbor query is performed using hierarchical traversal and bounding-box pruning, after which the result is verified against a brute-force nearest-neighbor search. The printed diagnostics demonstrate that only a subset of nodes and point distances must be examined to obtain the exact nearest neighbor. This illustrates the fundamental advantage of hierarchical spatial indexing: reducing geometric search cost through recursive subdivision and conservative pruning.

```rust
// Program 21.6.3: Balanced k-d Tree Nearest-Neighbor Search with Bounding-Box Pruning
//
// Problem statement:
// Given a finite point set P in R^d, construct a balanced k-d tree by recursively
// splitting the point set along the coordinate direction with largest spread,
// as in Equation (21.6.16). Each node stores an axis-aligned bounding box,
// and nearest-neighbor search maintains the best squared distance found so far,
// as in Equation (21.6.17). A subtree is safely pruned only when its bounding-box
// distance is strictly larger than the current best squared distance, following
// Equation (21.6.18).

#[derive(Clone, Debug)]
struct Point {
    coords: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Neighbor {
    index: usize,
    squared_distance: f64,
}

#[derive(Clone, Debug)]
struct Aabb {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[derive(Clone, Debug)]
struct KdNode {
    bounds: Aabb,
    point_index: usize,
    split_axis: usize,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

#[derive(Debug)]
struct KdTree {
    points: Vec<Point>,
    root: Option<Box<KdNode>>,
    dimension: usize,
}

#[derive(Debug)]
struct SearchStats {
    nodes_visited: usize,
    nodes_pruned: usize,
    point_tests: usize,
}

impl Point {
    fn new(coords: Vec<f64>) -> Self {
        Self { coords }
    }

    fn dimension(&self) -> usize {
        self.coords.len()
    }
}

impl Aabb {
    fn from_points(points: &[Point], indices: &[usize]) -> Self {
        let dimension = points[indices[0]].dimension();

        let mut lower = vec![f64::INFINITY; dimension];
        let mut upper = vec![f64::NEG_INFINITY; dimension];

        for &idx in indices {
            for axis in 0..dimension {
                lower[axis] = lower[axis].min(points[idx].coords[axis]);
                upper[axis] = upper[axis].max(points[idx].coords[axis]);
            }
        }

        Self { lower, upper }
    }

    fn squared_distance_to_point(&self, query: &Point) -> f64 {
        let mut total = 0.0;

        for axis in 0..query.dimension() {
            let x = query.coords[axis];

            let delta = if x < self.lower[axis] {
                self.lower[axis] - x
            } else if x > self.upper[axis] {
                x - self.upper[axis]
            } else {
                0.0
            };

            total += delta * delta;
        }

        total
    }
}

impl KdTree {
    fn new(points: Vec<Point>) -> Self {
        assert!(!points.is_empty(), "point set must not be empty");

        let dimension = points[0].dimension();
        assert!(dimension > 0, "dimension must be positive");

        for p in &points {
            assert_eq!(
                p.dimension(),
                dimension,
                "all points must have the same dimension"
            );
        }

        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = Self::build_recursive(&points, &mut indices);

        Self {
            points,
            root,
            dimension,
        }
    }

    fn build_recursive(points: &[Point], indices: &mut [usize]) -> Option<Box<KdNode>> {
        if indices.is_empty() {
            return None;
        }

        let bounds = Aabb::from_points(points, indices);
        let split_axis = largest_spread_axis(points, indices);

        indices.sort_by(|&a, &b| {
            points[a].coords[split_axis]
                .partial_cmp(&points[b].coords[split_axis])
                .unwrap()
                .then_with(|| a.cmp(&b))
        });

        let mid = indices.len() / 2;
        let point_index = indices[mid];

        let left = Self::build_recursive(points, &mut indices[..mid]);
        let right = Self::build_recursive(points, &mut indices[mid + 1..]);

        Some(Box::new(KdNode {
            bounds,
            point_index,
            split_axis,
            left,
            right,
        }))
    }

    fn nearest_neighbor(&self, query: &Point) -> Option<(Neighbor, SearchStats)> {
        assert_eq!(
            query.dimension(),
            self.dimension,
            "query dimension must match tree dimension"
        );

        let mut best: Option<Neighbor> = None;
        let mut stats = SearchStats {
            nodes_visited: 0,
            nodes_pruned: 0,
            point_tests: 0,
        };

        if let Some(root) = &self.root {
            self.search_recursive(root, query, &mut best, &mut stats);
        }

        best.map(|neighbor| (neighbor, stats))
    }

    fn search_recursive(
        &self,
        node: &KdNode,
        query: &Point,
        best: &mut Option<Neighbor>,
        stats: &mut SearchStats,
    ) {
        stats.nodes_visited += 1;

        let bound_dist2 = node.bounds.squared_distance_to_point(query);

        if let Some(current_best) = best {
            if bound_dist2 > current_best.squared_distance {
                stats.nodes_pruned += 1;
                return;
            }
        }

        stats.point_tests += 1;

        let point_dist2 = squared_distance(&self.points[node.point_index], query);

        match best {
            Some(current_best) => {
                if point_dist2 < current_best.squared_distance
                    || (point_dist2 == current_best.squared_distance
                        && node.point_index < current_best.index)
                {
                    *current_best = Neighbor {
                        index: node.point_index,
                        squared_distance: point_dist2,
                    };
                }
            }
            None => {
                *best = Some(Neighbor {
                    index: node.point_index,
                    squared_distance: point_dist2,
                });
            }
        }

        let axis = node.split_axis;
        let query_value = query.coords[axis];
        let node_value = self.points[node.point_index].coords[axis];

        let (near_child, far_child) = if query_value <= node_value {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(child) = near_child {
            self.search_recursive(child, query, best, stats);
        }

        if let Some(child) = far_child {
            self.search_recursive(child, query, best, stats);
        }
    }
}

fn largest_spread_axis(points: &[Point], indices: &[usize]) -> usize {
    let dimension = points[indices[0]].dimension();

    let mut best_axis = 0;
    let mut best_spread = f64::NEG_INFINITY;

    for axis in 0..dimension {
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;

        for &idx in indices {
            let value = points[idx].coords[axis];
            min_value = min_value.min(value);
            max_value = max_value.max(value);
        }

        let spread = max_value - min_value;

        if spread > best_spread {
            best_spread = spread;
            best_axis = axis;
        }
    }

    best_axis
}

fn squared_distance(a: &Point, b: &Point) -> f64 {
    assert_eq!(
        a.dimension(),
        b.dimension(),
        "points must have the same dimension"
    );

    a.coords
        .iter()
        .zip(b.coords.iter())
        .map(|(ai, bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum()
}

fn brute_force_nearest_neighbor(points: &[Point], query: &Point) -> Neighbor {
    points
        .iter()
        .enumerate()
        .map(|(index, p)| Neighbor {
            index,
            squared_distance: squared_distance(p, query),
        })
        .min_by(|a, b| {
            a.squared_distance
                .partial_cmp(&b.squared_distance)
                .unwrap()
                .then_with(|| a.index.cmp(&b.index))
        })
        .unwrap()
}

fn print_point(label: &str, point: &Point) {
    print!("{label}(");
    for (i, value) in point.coords.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{value:.6}");
    }
    println!(")");
}

fn print_neighbor(title: &str, points: &[Point], neighbor: &Neighbor) {
    println!("{title}");
    println!("{}", "-".repeat(title.len()));
    println!("index              = {}", neighbor.index);
    println!("squared distance   = {:.6}", neighbor.squared_distance);
    println!("distance           = {:.6}", neighbor.squared_distance.sqrt());
    print_point("point              = ", &points[neighbor.index]);
    println!();
}

fn main() {
    let points = vec![
        Point::new(vec![0.0, 0.0]),
        Point::new(vec![1.0, 2.0]),
        Point::new(vec![2.0, 1.0]),
        Point::new(vec![3.0, 3.0]),
        Point::new(vec![4.0, 2.0]),
        Point::new(vec![5.0, 4.0]),
        Point::new(vec![2.5, 2.5]),
        Point::new(vec![7.0, 7.0]),
        Point::new(vec![8.0, 7.5]),
        Point::new(vec![-1.0, -0.5]),
        Point::new(vec![6.0, 1.0]),
        Point::new(vec![3.5, 4.5]),
    ];

    let query = Point::new(vec![2.2, 2.0]);

    println!("Balanced k-d Tree Nearest-Neighbor Search");
    println!("==========================================\n");

    println!("Point Set");
    println!("---------");
    for (i, p) in points.iter().enumerate() {
        print!("p_{:<2} = ", i);
        print_point("", p);
    }
    println!();

    println!("Query Point");
    println!("-----------");
    print_point("x = ", &query);
    println!();

    let tree = KdTree::new(points);

    let (kd_neighbor, stats) = tree.nearest_neighbor(&query).unwrap();
    let brute_force_neighbor = brute_force_nearest_neighbor(&tree.points, &query);

    print_neighbor("k-d Tree Nearest Neighbor", &tree.points, &kd_neighbor);
    print_neighbor(
        "Brute-Force Verification Neighbor",
        &tree.points,
        &brute_force_neighbor,
    );

    println!("Search Diagnostics");
    println!("------------------");
    println!("tree dimension       = {}", tree.dimension);
    println!("number of points     = {}", tree.points.len());
    println!("nodes visited        = {}", stats.nodes_visited);
    println!("nodes pruned         = {}", stats.nodes_pruned);
    println!("point tests          = {}", stats.point_tests);
    println!(
        "agrees with brute force = {}",
        kd_neighbor.index == brute_force_neighbor.index
            && kd_neighbor.squared_distance == brute_force_neighbor.squared_distance
    );
}
```

Program 21.6.3 demonstrates how hierarchical spatial subdivision can dramatically reduce the number of candidate points examined during exact nearest-neighbor search. By recursively partitioning the point set into geometrically localized subsets and enclosing each subtree within an axis-aligned bounding box, the k-d tree transforms nearest-neighbor search from a global scanning problem into a localized hierarchical traversal problem.

The implementation illustrates the central role of conservative pruning in tree-based geometric search. Bounding-box distance evaluation provides a mathematically safe mechanism for eliminating entire subtrees whenever their minimum possible distance from the query point exceeds the current best candidate distance. As a result, only a fraction of the stored points must typically be examined in low-dimensional and reasonably well-distributed point clouds.

The example also highlights the broader computational philosophy underlying tree-based spatial indexing. The k-d tree combines recursive subdivision, geometric locality, and exact distance verification into a unified search framework. Although the present implementation focuses specifically on nearest-neighbor search for point clouds, the same principles extend naturally to ball trees, octrees, quadtrees, and bounding volume hierarchies. These structures differ primarily in how they subdivide space and represent bounding geometry, but all rely fundamentally on the same strategy of hierarchical candidate reduction followed by exact local geometric evaluation. The program therefore provides both a practical nearest-neighbor implementation and a conceptual foundation for more advanced hierarchical spatial-search structures used throughout scientific computing and computational geometry.

## 21.6.4. High-Dimensional Effects and Approximate Search

Spatial search becomes fundamentally harder as the dimension increases. Many tree-based methods rely on the idea that geometric pruning removes large parts of the data set. In high dimensions, this becomes less effective because distances tend to concentrate. For example, if distances from a query point to many data points are nearly equal, then the distinction between nearest and non-nearest points becomes weak. In such cases, exact nearest-neighbor search may require inspecting a large fraction of the data set, even when a sophisticated tree is used.

One way to express this issue is through the contrast ratio:

$$
\gamma(x)
=
\frac{d_2(x)}{d_1(x)}
\tag{21.6.26}
$$

where $d_1(x)$ and $d_2(x)$ are the distances from $x$ to its first and second nearest neighbors. If $\gamma(x)\approx 1$, then the nearest neighbor is weakly distinguished. Small perturbations, noise, or floating-point effects may alter the identity of the nearest point. In high-dimensional data, such near-ties become more common, and the cost of certifying the exact nearest neighbor can increase substantially.

This difficulty is not only computational but also interpretive. In a low-dimensional geometric mesh, nearest-neighbor search usually has clear spatial meaning. In a high-dimensional feature space, the nearest point under Euclidean distance may not be scientifically meaningful unless the metric has been carefully chosen. Therefore, high-dimensional search often requires metric design, dimensional reduction, approximation, or domain-specific structure.

Approximate nearest-neighbor methods relax exactness in exchange for speed. A common guarantee is the $(1+\varepsilon)$-approximation condition:

$$
\|x-\widetilde{p}\|_2
\le
(1+\varepsilon)
\min_{p\in P}
\|x-p\|_2
\tag{21.6.27}
$$

For many scientific and machine-learning workflows, this is sufficient because the input data themselves may contain noise or because the downstream approximation is not sensitive to the exact identity of the nearest point. However, in conservative remapping, mesh validity checks, or certified geometric algorithms, approximate search may be unacceptable unless it is followed by an exact verification stage.

Fixed-radius search has different behavior. Instead of asking for the single closest point, it asks for all points satisfying $\|x-p_j\|_2\le r$. This query is often more meaningful in physical simulations because the radius corresponds to a support radius, interaction length, or smoothing scale. The result may contain many points, so the output size itself contributes to the complexity $\Omega(|\mathcal{N}_r(x)|)$. No algorithm can report the neighborhood faster than the size of the neighborhood it must output. For this reason, performance analysis should distinguish candidate-generation cost from output cost.

Recent work on point-cloud search taxonomies and high-dimensional nearest-neighbor behavior supports a more nuanced view of spatial indexing: brute force, grids, $k$-d trees, octrees, BVHs, and approximate methods are not ordered on a single scale from simple to advanced. They solve different query regimes. The correct method depends on dimension, distribution, required exactness, query radius, update frequency, hardware, and the numerical meaning of the query result (Teuscher et al., 2025; Ting et al., 2024).

## 21.6.5. Complexity, Robustness, and Rust Implementation Notes

The baseline costs of common spatial search methods can be summarized as follows. Brute-force nearest-neighbor search costs $O(Nd)$ per query. Uniform-grid radius search costs approximately $O(B(x)+|\mathcal{N}_r(x)|)$, where $B(x)$ denotes the number of candidate points in the inspected buckets. A balanced $k$-d tree typically has build cost $O(N\log N)$ and average low-dimensional query behavior often close to $O(\log N)$, although worst-case or high-dimensional behavior may be much worse. Octree and BVH costs depend strongly on subdivision quality, object distribution, bounding overlap, and traversal order. Approximate methods trade exactness for speed and must be judged by both computational cost and approximation error.

Robustness in spatial search has a different character from robustness in orientation predicates. The main issue is usually not the sign of a determinant, but consistent classification near boundaries. For example, a point lying exactly on a grid-cell boundary may be assigned to one of two adjacent cells. A query sphere may pass exactly through a point $\|x-p_j\|_2=r$.\
\
A bounding-box pruning test may be exactly tight:

$$\operatorname{dist}(x,B)^2=\rho^2\tag{21.6.28}$$

Such cases require deterministic conventions. For exact fixed-radius search, the comparison should be formulated consistently, for example using squared distances $\|x-p_j\|_2^2\le r^2$. For pruning, one must ensure that a subtree is discarded only when it cannot contain a valid answer. Thus, a safe pruning condition is strict:

$$\operatorname{dist}(x,B)^2>\rho^2 \tag{21.6.29}$$

not merely greater than or equal, unless equality has a documented tie-breaking policy.

From a Rust implementation perspective, spatial indexes should separate geometry, payload, metric, and traversal policy. A useful design structure is:

$$
\text{points and payloads}
\longrightarrow
\text{index construction}
\longrightarrow
\text{candidate generation}\\
\longrightarrow
\text{exact local verification}
\longrightarrow
\text{query result}
\tag{21.6.30}
$$

The point storage should usually be contiguous, for example a vector of points or structure-of-arrays layout when vectorization is important. The payload should be stored by index rather than duplicated in tree nodes. Query functions should return stable point or object indices, not borrowed references that complicate later mutation.

For grids and voxel hashes, integer cell coordinates should be explicit:

$$(i_1,\ldots,i_d)=g(p) \tag{21.6.31}$$

The implementation should define how negative coordinates, domain offsets, and boundary points are handled. For tree structures, nodes should ideally be stored in arrays:

$$\text{node}_k=(\text{bounds},\text{children},\text{range}) \tag{21.6.32}$$

where children are integer indices and ranges refer to contiguous spans of points or primitives. This layout is usually more cache-friendly and easier to validate than pointer-rich recursive structures.

The main implementation principle is that spatial search should produce candidates, not silently replace exact geometry. For example, a BVH may quickly identify triangles whose bounding boxes intersect a query box, but exact triangle intersection must still be performed afterward. A grid may identify candidate neighbors, but squared-distance checks must still filter the final radius neighborhood. This two-stage structure,

$$\text{coarse spatial pruning}\quad+\quad\text{exact local predicate}\tag{21.6.33}$$

is the safest way to combine performance with correctness. It also matches the broader philosophy of this chapter: use data structures to reduce work, but preserve reliable geometric decisions at the final classification stage.

### Rust Implementation

Following the discussion in Sections 21.6.4 and 21.6.5 on high-dimensional effects, approximate nearest-neighbor search, and robustness considerations in spatial indexing, Program 21.6.4 provides a practical implementation for analyzing nearest-neighbor contrast ratios and validating approximate search results through exact geometric verification. The program demonstrates how nearest-neighbor structure changes in high-dimensional spaces by computing the contrast ratio introduced in Equation (21.6.26), which measures the separation between the nearest and second-nearest neighbors of a query point. It also illustrates the practical role of approximate search by testing whether a sampled approximate neighbor satisfies the $(1+\varepsilon)$-approximation condition of Equation (21.6.27). Rather than focusing solely on asymptotic acceleration, the implementation emphasizes the broader computational philosophy discussed in Section 21.6.5: spatial indexing should first generate candidate points efficiently and then apply exact local geometric verification to preserve correctness. The program therefore combines brute-force reference search, approximate candidate sampling, contrast-ratio analysis, and exact verification into a unified framework for studying the behavior of spatial search in moderately high-dimensional spaces.

At the core of the implementation is the `Point` structure, which stores coordinates in a dynamically sized vector and therefore supports arbitrary-dimensional feature spaces. The `Neighbor` structure stores the index of a candidate point together with its squared Euclidean distance from the query point, thereby preserving both geometric and indexing information during search operations. The `ApproximateSearchReport` structure aggregates the complete diagnostic output of the program, including the exact nearest and second-nearest neighbors, the sampled approximate neighbor, the contrast ratio, the approximation ratio, the approximation-validity flag, and the candidate counts examined during search. This organization separates geometric computation from diagnostic interpretation and reflects the implementation philosophy described in Equation (21.6.30).

The `squared_distance` function implements Euclidean distance evaluation using squared norms rather than explicit square roots. This follows the same computational principle used throughout Section 21.6: nearest-neighbor comparisons and radius checks may be performed entirely using squared distances without altering geometric ordering. The function iterates coordinate-wise through the point vectors, accumulates squared coordinate differences, and returns the resulting squared norm. This formulation avoids unnecessary square-root evaluations while remaining fully consistent with the exact geometric predicates required for reliable spatial search.

The `sorted_distances` function computes and sorts the complete set of distances between the query point and all stored points. The resulting ordered neighbor list forms the basis for exact nearest-neighbor analysis and provides the geometric reference against which approximate methods are validated. Although this approach has brute-force complexity $O(Nd)$, it provides an exact benchmark implementation and ensures that all later approximation checks remain geometrically reliable.

The `exact_two_nearest` function extracts the first and second nearest neighbors from the sorted distance list. These two neighbors are required to evaluate the contrast ratio introduced in Equation (21.6.26). The nearest-neighbor distance $d_1(x)$ and second-nearest-neighbor distance $d_2(x)$ together provide a quantitative measure of how strongly distinguished the nearest neighbor is from competing candidates. When the ratio approaches one, the nearest-neighbor identity becomes increasingly unstable under perturbation, noise, or floating-point effects.

The `contrast_ratio` function evaluates the quantity $\gamma(x)=\frac{d_2(x)}{d_1(x)},$ which serves as a diagnostic measure of nearest-neighbor separation in high-dimensional spaces. The function converts the stored squared distances into Euclidean distances before computing the ratio. A large value of $\gamma(x)$ indicates strong separation between the first and second nearest neighbors, whereas values close to one indicate weak geometric distinction. This behavior reflects the distance-concentration effects discussed in Section 21.6.4, where many distances become nearly equal in high-dimensional spaces.

The `approximate_nearest_from_sample` function demonstrates a simplified approximate nearest-neighbor strategy based on candidate subsampling. Rather than testing every point in the data set, the function examines only every $k^\text{th}$ point according to a prescribed stride length. The best candidate from this reduced subset is returned as the approximate nearest neighbor. Although intentionally simple, this implementation illustrates the fundamental idea underlying approximate search methods: reducing candidate evaluations in exchange for possible loss of exactness. The function also reports the number of sampled candidates examined, making it possible to compare approximate-search cost with full brute-force search.

The `verify_approximation` function implements the $(1+\varepsilon)$-approximation test introduced in Equation (21.6.27). After computing the exact and approximate Euclidean distances, the function evaluates the approximation ratio

$$\frac{\|x-\widetilde{p}\|_2}{\min{p\in P}\|x-p\|_2},$$

and checks whether this quantity is bounded above by $1+\varepsilon$. This explicit verification stage reflects the implementation philosophy emphasized in Section 21.6.5: approximate search should not silently replace exact geometry without a clearly defined validation criterion.

The `analyze_query` function combines exact nearest-neighbor analysis, contrast-ratio evaluation, approximate search, and approximation verification into a single high-level workflow. It first computes the exact nearest and second-nearest neighbors, then evaluates the contrast ratio, performs approximate search using candidate sampling, and finally verifies whether the approximate result satisfies the prescribed approximation bound. The resulting `ApproximateSearchReport` structure consolidates all geometric and diagnostic information associated with the query.

The functions `generate_high_dimensional_points` and `generate_query` construct a synthetic high-dimensional data set and query point for testing purposes. Rather than relying on random-number generation, the implementation uses deterministic trigonometric expressions to produce reproducible coordinate values. This ensures that the same geometric configuration is generated consistently across executions while still producing nontrivial high-dimensional structure.

The printing functions separate output formatting from geometric computation. Functions such as `print_point_prefix`, `print_neighbor`, and `print_report` present the nearest-neighbor diagnostics, approximation statistics, and contrast-ratio measurements in a structured form suitable for direct interpretation. The diagnostic output exposes the relationship between exact and approximate search while also illustrating how many candidate points were examined in each search regime.

The `main` function serves to demonstrate high-dimensional nearest-neighbor analysis and approximate-search verification using a synthetic point cloud in $\mathbb{R}^{32}$. It begins by constructing a moderately sized high-dimensional data set together with a query point and user-defined approximation tolerance $\varepsilon$. The program then computes the exact nearest and second-nearest neighbors, evaluates the contrast ratio, performs approximate search using candidate subsampling, and verifies whether the resulting approximate neighbor satisfies the prescribed approximation bound. Finally, the program prints an interpretive summary explaining whether the nearest-neighbor structure is strongly or weakly separated and whether the approximate result satisfies the required geometric accuracy condition. This workflow illustrates the combined computational and interpretive challenges associated with high-dimensional spatial search.

```rust
// Program 21.6.4: High-Dimensional Nearest-Neighbor Contrast and Approximate
// Search Verification
//
// Problem statement:
// Given a point set P in R^d and a query point x, compute the exact nearest
// and second-nearest neighbors, evaluate the contrast ratio gamma(x) = d_2/d_1
// from Equation (21.6.26), and test whether an approximate nearest neighbor
// satisfies the (1 + epsilon)-approximation condition in Equation (21.6.27).
// The program also demonstrates the implementation principle of separating
// candidate generation from exact local verification, as described in
// Equations (21.6.30) and (21.6.33).

#[derive(Clone, Debug)]
struct Point {
    coords: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Neighbor {
    index: usize,
    squared_distance: f64,
}

#[derive(Debug)]
struct ApproximateSearchReport {
    exact_nearest: Neighbor,
    second_nearest: Neighbor,
    approximate: Neighbor,
    contrast_ratio: f64,
    epsilon: f64,
    approximation_ratio: f64,
    satisfies_bound: bool,
    candidates_examined: usize,
    full_points_examined: usize,
}

impl Point {
    fn new(coords: Vec<f64>) -> Self {
        Self { coords }
    }

    fn dimension(&self) -> usize {
        self.coords.len()
    }
}

fn squared_distance(a: &Point, b: &Point) -> f64 {
    assert_eq!(
        a.dimension(),
        b.dimension(),
        "points must have the same dimension"
    );

    a.coords
        .iter()
        .zip(b.coords.iter())
        .map(|(ai, bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum()
}

fn sorted_distances(points: &[Point], query: &Point) -> Vec<Neighbor> {
    let mut distances: Vec<Neighbor> = points
        .iter()
        .enumerate()
        .map(|(index, p)| Neighbor {
            index,
            squared_distance: squared_distance(p, query),
        })
        .collect();

    distances.sort_by(|a, b| {
        a.squared_distance
            .partial_cmp(&b.squared_distance)
            .unwrap()
            .then_with(|| a.index.cmp(&b.index))
    });

    distances
}

fn exact_two_nearest(points: &[Point], query: &Point) -> (Neighbor, Neighbor) {
    assert!(
        points.len() >= 2,
        "at least two points are required to compute the contrast ratio"
    );

    let distances = sorted_distances(points, query);

    (distances[0].clone(), distances[1].clone())
}

fn contrast_ratio(first: &Neighbor, second: &Neighbor) -> f64 {
    let d1 = first.squared_distance.sqrt();
    let d2 = second.squared_distance.sqrt();

    if d1 == 0.0 {
        f64::INFINITY
    } else {
        d2 / d1
    }
}

fn approximate_nearest_from_sample(
    points: &[Point],
    query: &Point,
    sample_stride: usize,
) -> (Neighbor, usize) {
    assert!(!points.is_empty(), "point set must not be empty");
    assert!(sample_stride > 0, "sample stride must be positive");

    let mut best: Option<Neighbor> = None;
    let mut candidates_examined = 0usize;

    for index in (0..points.len()).step_by(sample_stride) {
        candidates_examined += 1;

        let dist2 = squared_distance(&points[index], query);

        match &mut best {
            Some(current_best) => {
                if dist2 < current_best.squared_distance
                    || (dist2 == current_best.squared_distance && index < current_best.index)
                {
                    *current_best = Neighbor {
                        index,
                        squared_distance: dist2,
                    };
                }
            }
            None => {
                best = Some(Neighbor {
                    index,
                    squared_distance: dist2,
                });
            }
        }
    }

    (best.unwrap(), candidates_examined)
}

fn verify_approximation(
    exact: &Neighbor,
    approximate: &Neighbor,
    epsilon: f64,
) -> (f64, bool) {
    assert!(epsilon >= 0.0, "epsilon must be nonnegative");

    let exact_distance = exact.squared_distance.sqrt();
    let approximate_distance = approximate.squared_distance.sqrt();

    let ratio = if exact_distance == 0.0 {
        if approximate_distance == 0.0 {
            1.0
        } else {
            f64::INFINITY
        }
    } else {
        approximate_distance / exact_distance
    };

    let satisfies_bound = ratio <= 1.0 + epsilon;

    (ratio, satisfies_bound)
}

fn analyze_query(
    points: &[Point],
    query: &Point,
    epsilon: f64,
    sample_stride: usize,
) -> ApproximateSearchReport {
    let (exact_nearest, second_nearest) = exact_two_nearest(points, query);
    let gamma = contrast_ratio(&exact_nearest, &second_nearest);

    let (approximate, candidates_examined) =
        approximate_nearest_from_sample(points, query, sample_stride);

    let (approximation_ratio, satisfies_bound) =
        verify_approximation(&exact_nearest, &approximate, epsilon);

    ApproximateSearchReport {
        exact_nearest,
        second_nearest,
        approximate,
        contrast_ratio: gamma,
        epsilon,
        approximation_ratio,
        satisfies_bound,
        candidates_examined,
        full_points_examined: points.len(),
    }
}

fn generate_high_dimensional_points(n: usize, dimension: usize) -> Vec<Point> {
    assert!(n > 0, "number of points must be positive");
    assert!(dimension > 0, "dimension must be positive");

    let mut points = Vec::with_capacity(n);

    for i in 0..n {
        let mut coords = Vec::with_capacity(dimension);

        for j in 0..dimension {
            let angle_1 = 0.173 * (i as f64 + 1.0) * (j as f64 + 1.0);
            let angle_2 = 0.097 * (i as f64 + 2.0) * (j as f64 + 3.0);

            let value = angle_1.sin() + 0.5 * angle_2.cos();
            coords.push(value);
        }

        points.push(Point::new(coords));
    }

    points
}

fn generate_query(dimension: usize) -> Point {
    let mut coords = Vec::with_capacity(dimension);

    for j in 0..dimension {
        let value = (0.211 * (j as f64 + 1.0)).sin()
            + 0.25 * (0.137 * (j as f64 + 4.0)).cos();
        coords.push(value);
    }

    Point::new(coords)
}

fn print_point_prefix(label: &str, point: &Point, max_entries: usize) {
    print!("{label}(");

    let entries = point.dimension().min(max_entries);

    for i in 0..entries {
        if i > 0 {
            print!(", ");
        }
        print!("{:.6}", point.coords[i]);
    }

    if point.dimension() > max_entries {
        print!(", ...");
    }

    println!(")");
}

fn print_neighbor(label: &str, neighbor: &Neighbor) {
    println!("{label}");
    println!("{}", "-".repeat(label.len()));
    println!("index              = {}", neighbor.index);
    println!("squared distance   = {:.10}", neighbor.squared_distance);
    println!("distance           = {:.10}", neighbor.squared_distance.sqrt());
    println!();
}

fn print_report(report: &ApproximateSearchReport) {
    print_neighbor("Exact Nearest Neighbor", &report.exact_nearest);
    print_neighbor("Exact Second-Nearest Neighbor", &report.second_nearest);
    print_neighbor("Approximate Neighbor from Sampled Candidates", &report.approximate);

    println!("High-Dimensional and Approximation Diagnostics");
    println!("----------------------------------------------");
    println!("contrast ratio gamma       = {:.10}", report.contrast_ratio);
    println!("epsilon                    = {:.10}", report.epsilon);
    println!(
        "allowed ratio 1 + epsilon  = {:.10}",
        1.0 + report.epsilon
    );
    println!(
        "approximation ratio        = {:.10}",
        report.approximation_ratio
    );
    println!(
        "satisfies approximation    = {}",
        report.satisfies_bound
    );
    println!(
        "sampled candidates tested  = {}",
        report.candidates_examined
    );
    println!(
        "full brute-force tests     = {}",
        report.full_points_examined
    );
}

fn main() {
    let n = 128usize;
    let dimension = 32usize;
    let epsilon = 0.10;
    let sample_stride = 3usize;

    let points = generate_high_dimensional_points(n, dimension);
    let query = generate_query(dimension);

    println!("High-Dimensional Effects and Approximate Search");
    println!("================================================\n");

    println!("Problem Setup");
    println!("-------------");
    println!("number of points N      = {}", n);
    println!("dimension d             = {}", dimension);
    println!("epsilon                 = {:.6}", epsilon);
    println!("candidate sample stride = {}", sample_stride);
    println!();

    println!("Query Point Prefix");
    println!("------------------");
    print_point_prefix("x = ", &query, 8);
    println!();

    let report = analyze_query(&points, &query, epsilon, sample_stride);
    print_report(&report);

    println!();
    println!("Interpretation");
    println!("--------------");
    if report.contrast_ratio < 1.05 {
        println!(
            "The contrast ratio is close to one, so the nearest and second-nearest \
             neighbors are weakly separated."
        );
    } else {
        println!(
            "The contrast ratio is noticeably larger than one, so the nearest \
             neighbor is more clearly distinguished from the second nearest."
        );
    }

    if report.satisfies_bound {
        println!(
            "The sampled approximate neighbor satisfies the prescribed \
             (1 + epsilon)-approximation condition."
        );
    } else {
        println!(
            "The sampled approximate neighbor does not satisfy the prescribed \
             (1 + epsilon)-approximation condition."
        );
    }
}
```

Program 21.6.4 demonstrates how high-dimensional geometry alters the behavior of nearest-neighbor search and motivates the use of approximate search methods. By computing both the exact nearest-neighbor structure and the contrast ratio $\gamma(x)$, the implementation reveals how strongly or weakly distinguished the nearest neighbor is from nearby competing candidates. In high-dimensional settings, this distinction often becomes less pronounced, making exact nearest-neighbor certification increasingly expensive and sometimes less meaningful from a geometric perspective.

The example also illustrates the practical role of approximate search. Instead of evaluating every point in the data set, the approximate search stage reduces computational effort by testing only a subset of candidate points. The resulting approximation is then verified explicitly against the $(1+\varepsilon)$-approximation condition, thereby separating candidate generation from exact geometric validation. This two-stage structure reflects the implementation philosophy emphasized throughout Sections 21.6.4 and 21.6.5: acceleration structures and approximation techniques should reduce work, but exact geometric predicates should remain responsible for final correctness verification.

The program further demonstrates that performance analysis in spatial search must consider more than asymptotic complexity alone. Approximation quality, dimensionality, contrast ratios, candidate counts, and robustness all influence the practical effectiveness of a search strategy. The implementation therefore provides both a computational example of approximate nearest-neighbor analysis and a conceptual foundation for more advanced methods such as randomized projection trees, locality-sensitive hashing, graph-based approximate search, and hybrid exact-approximate verification pipelines used in modern high-dimensional scientific computing and machine-learning applications.

# 21.7. Meshes, Barycentric Coordinates, and Point Location

Meshes are the geometric structures through which continuous domains become finite computational objects. In a finite element method, a mesh determines the elements on which local basis functions are defined. In a finite volume method, it determines the control volumes across which fluxes are balanced. In interpolation, remapping, visualization, particle-to-mesh transfer, and scattered-data reconstruction, it determines where data live and how values are transported between locations. A mesh is therefore not merely a collection of drawn cells. It is a coupled geometric and topological object whose validity affects approximation error, conservation, matrix conditioning, and solver reliability. Modern mesh-generation and barycentric-coordinate literature emphasizes that element quality, boundary fidelity, point containment, and stable coordinate evaluation are essential parts of scientific-computing geometry (Lei et al., 2024; Zou et al., 2024; Ji et al., 2024; Dieci, Difonzo and Sukumar, 2024; Fuda and Hormann, 2024; Romano et al., 2025).

## 21.7.1. Mesh Topology, Geometry, and Orientation

A mesh over a domain $\Omega\subset\mathbb{R}^d$ is a finite collection of cells,

$$\mathcal{T}_h=\{K_1,K_2,\ldots,K_N\} \tag{21.7.1}$$

together with incidence and adjacency relations among their lower-dimensional entities. In two dimensions, the cells may be triangles, quadrilaterals, or general polygons. In three dimensions, they may be tetrahedra, hexahedra, prisms, pyramids, or general polyhedra. The mesh approximates the domain in the sense that,

$$
\overline{\Omega}_h
=
\bigcup_{K\in\mathcal{T}_h}
\overline{K} 
\tag{21.7.2}
$$

where $\Omega_h$ is the discrete computational domain. If the mesh exactly represents $\Omega$, then $\Omega_h=\Omega.$ More commonly, $\Omega_h$ is a polygonal, polyhedral, curved, or high-order approximation of the physical domain.

The topology of a mesh is encoded by its vertices, edges, faces, and cells. If $V$, $E$, $F$, and $C$ denote the sets of vertices, edges, faces, and cells, then incidence relations specify which vertices belong to an edge, which edges bound a face, and which faces bound a cell. In a simplicial three-dimensional mesh, for example, a tetrahedron may be represented by four vertex indices,

$$K=(i_0,i_1,i_2,i_3)\tag{21.7.3}$$

where the physical coordinates are $v_{i_0},v_{i_1},v_{i_2},v_{i_3}\in\mathbb{R}^3$. Neighbor relations then specify which tetrahedron lies across each face. This separation between coordinates and connectivity is essential for robust implementation.

A conforming mesh satisfies the condition that the intersection of two distinct closed cells is either empty or a common lower-dimensional entity:

$$
\overline{K_i}\cap \overline{K_j}
\in
\left\{
\varnothing,
\text{ common vertex},
\text{ common edge},
\text{ common face}
\right\}
\tag{21.7.4}
$$

In two dimensions, two triangles may share an edge or a vertex, but one triangle edge should not terminate in the middle of another triangle edge unless the mesh is intentionally nonconforming. In three dimensions, two tetrahedra may share a full face, an edge, or a vertex, but partial face overlaps require additional constraints or mortar-like treatment. Nonconforming, cut-cell, and hybrid meshes relax these restrictions, but they then require more sophisticated geometric intersection and adjacency logic.

Orientation is equally important. For a triangular cell $K=(v_0,v_1,v_2)$, the signed area is:

$$A_{\mathrm{signed}}(K) = \frac{1}{2}\operatorname{orient2d}(v_0,v_1,v_2)\tag{21.7.5}$$

A valid triangle must satisfy:

$$A(K)=\frac{1}{2}\left|\operatorname{orient2d}(v_0,v_1,v_2)\right|>0 \tag{21.7.6}$$

For a tetrahedron $K=(v_0,v_1,v_2,v_3)$, the signed volume is:

$$V_{\mathrm{signed}}(K) = \frac{1}{6}\operatorname{orient3d}(v_0,v_1,v_2,v_3) \tag{21.7.7}$$

and a valid nondegenerate tetrahedron must satisfy,

$$
V(K)
=
\frac{1}{6}
\left|
\operatorname{orient3d}(v_0,v_1,v_2,v_3)
\right|
> 0
\tag{21.7.8}
$$

If the signed area or signed volume has the wrong sign relative to the convention used by the code, local normal directions, Jacobian determinants, and element matrices may be assembled incorrectly. If it is near zero, the element is geometrically degenerate or nearly degenerate, which usually leads to poor conditioning.

The geometric map from a reference cell to a physical cell makes this connection explicit. For a simplex $K=\operatorname{conv}{v_0,\ldots,v_d}$, define the affine map:

$$x=F_K(\widehat{x})=v_0+B_K\widehat{x} \tag{21.7.9}$$

where,

$$
B_K
=
\begin{bmatrix}
v_1-v_0 & v_2-v_0 & \cdots & v_d-v_0
\end{bmatrix}
\tag{21.7.10}
$$

The determinant of $B_K$ controls the volume scaling:

$$
|K|
=
\frac{|\det B_K|}{d!}
\tag{21.7.11}
$$

Thus, element validity requires:

$$\det B_K\ne 0 \tag{21.7.12}$$

In finite element assembly, gradients transform by:

$$\nabla_x \phi = B_K^{-T}\nabla_{\widehat{x}}\widehat{\phi} \tag{21.7.13}$$

If $B_K$ is ill conditioned, gradients and stiffness contributions may become inaccurate or unstable. Mesh quality is therefore directly connected to the conditioning of local coordinate transformations.

## 21.7.2. Mesh Quality and Element Shape Measures

Mesh quality is a numerical property as much as a geometric one. A mesh with valid topology may still be unsuitable for computation if its elements are highly stretched, nearly degenerate, inverted, or poorly aligned with the solution features. Such elements can degrade interpolation accuracy, increase matrix condition numbers, reduce time-step stability, and cause nonlinear solvers to fail. Modern simulation-oriented meshing therefore treats quality control as a central objective rather than a cosmetic post-processing step (Lei et al., 2024; Zou et al., 2024; Ji et al., 2024; Qiu et al., 2026).

For a simplex $K$, let $h_K$ denote its diameter:

$$h_K=\max_{x,y\in K}\|x-y\|_2 \tag{21.7.14}$$

For a triangle or tetrahedron, this is the maximum edge length. Let $\rho_K$ denote the inradius, the radius of the largest ball contained in $K$, and let $R_K$ denote the circumradius, the radius of the circumscribed sphere. A common quality measure is the radius ratio:

$$q_K = C_d \frac{\rho_K}{R_K} \tag{21.7.15}$$

where $C_d$ is chosen so that $q_K=1$ for a regular simplex. In two dimensions, one often uses:

$$q_K=\frac{2r_K}{R_K} \tag{21.7.16}$$

where $r_K$ is the inradius and $R_K$ is the circumradius of a triangle. The value $q_K$ approaches zero as the triangle degenerates.

For a triangle with side lengths $a,b,c$, area $A_K$, and semiperimeter,

$$s=\frac{a+b+c}{2} \tag{21.7.17}$$

the inradius and circumradius are:

$$r_K=\frac{A_K}{s} \tag{21.7.18}$$

and

$$R_K=\frac{abc}{4A_K} \tag{21.7.19}$$

Thus,

$$
q_K
=
\frac{2r_K}{R_K}
=
\frac{8A_K^2}{sabc}
\tag{21.7.20}
$$

This formula makes the degeneracy clear: if $A_K\to 0$ while edge lengths remain nonzero, then $q_K\to 0$. Another important measure is the minimum angle:

$$\theta_{\min}(K)=\min_i \theta_i \tag{21.7.21}$$

Small angles are harmful for interpolation and stiffness-matrix conditioning. In two dimensions, shape-regular families of triangulations are often characterized by a lower bound:

$$\theta_{\min}(K)\ge \theta_0>0\qquad\text{for all }K\in\mathcal{T}_h \tag{21.7.22}$$

Equivalently, one may impose a bound on the aspect ratio. A representative simplex aspect ratio is:

$$\operatorname{AR}(K)=\frac{h_K}{\rho_K} \tag{21.7.23}$$

A family of meshes is shape regular if there exists a constant $C$, independent of mesh refinement, such that,

$$\frac{h_K}{\rho_K}\le C\qquad\text{for all }K\in\mathcal{T}_h \tag{21.7.24}$$

This condition prevents arbitrarily flat or needle-like elements as the mesh is refined.

The Jacobian matrix $B_K$ from equation (21.7.10) provides another way to quantify element quality. The condition number,

$$
\kappa(B_K)
=
\|B_K\|_2\,\|B_K^{-1}\|_2
\tag{21.7.25}
$$

measures distortion of the affine map from the reference simplex to the physical simplex. A large value of $\kappa(B_K)$ indicates a stretched or nearly singular element. Since finite element gradients involve $B_K^{-T}$, as in equation (21.7.13), the conditioning of $B_K$ directly affects numerical stability.

For high-order or curved elements, validity requires more than a positive affine volume. If the physical map is:

$$x=F_K(\widehat{x}) \tag{21.7.26}$$

with Jacobian matrix,

$$J_K(\widehat{x})=\frac{\partial F_K}{\partial \widehat{x}} \tag{21.7.27}$$

then element validity requires:

$$\det J_K(\widehat{x})>0\qquad\text{for all }\widehat{x}\in \widehat{K} \tag{21.7.28}$$

If $\det J_K$ becomes zero or changes sign inside the reference element, the curved element folds over itself. High-order tetrahedral optimization and NURBS-enhanced finite element mesh generation address precisely this issue: the mesh must approximate curved geometry accurately while preserving valid, well-conditioned elements (Zou et al., 2024; Ji et al., 2024).

### Rust Implementation

Following the discussion in Sections 21.7.1 and 21.7.2 on mesh topology, orientation, affine mappings, and element quality, Program 21.7.1 provides a practical implementation of triangle mesh validity and quality diagnostics for two-dimensional simplicial meshes. The program demonstrates how geometric and topological information combine to determine whether a mesh is suitable for scientific computation. Using signed orientation tests, affine element mappings, and classical shape-quality measures, the implementation evaluates whether individual triangles are consistently oriented, geometrically valid, well conditioned, and sufficiently regular for stable numerical approximation. In addition to detecting inverted and nearly degenerate elements, the program computes radius-ratio quality metrics, minimum angles, aspect ratios, and affine-map condition numbers, thereby connecting geometric mesh quality directly to the numerical stability issues discussed throughout Section 21.7. The implementation therefore illustrates the broader computational principle that a mesh is not merely a geometric visualization object, but a numerical structure whose quality directly influences interpolation accuracy, matrix conditioning, and solver robustness.

At the core of the implementation are the `Point2` and `Triangle` structures, which separate geometric coordinates from mesh connectivity. The `Point2` structure stores the physical coordinates of mesh vertices, while the `Triangle` structure stores only the integer indices of the three vertices associated with a triangular element. This separation between geometry and topology directly reflects the mesh representation philosophy described in Equation (21.7.3), where a simplex is represented through vertex connectivity rather than duplicated coordinate data. Such a design improves storage efficiency and simplifies adjacency and neighborhood operations in larger mesh-processing systems. The `TriangleQuality` structure stores all geometric and numerical diagnostics associated with a triangle, including area, orientation, radius-ratio quality, aspect ratio, minimum angle, and affine-map conditioning information.

The `orient2d` function implements the planar orientation predicate underlying the signed-area formulation of Equation (21.7.5). Given three vertices, the function evaluates the signed determinant associated with the oriented triangle. Positive values correspond to counterclockwise orientation, negative values correspond to clockwise orientation, and values near zero indicate geometric degeneracy. This orientation predicate forms the basis of many geometric algorithms because it determines whether a simplex has consistent orientation and nonzero area.

The `signed_area` function computes the signed triangle area directly from the orientation predicate. The returned quantity corresponds to the signed-area expression introduced in Equation (21.7.5), while the absolute value provides the geometric area appearing in Equation (21.7.6). The implementation distinguishes explicitly between orientation and geometric magnitude because both are numerically important. A triangle may possess positive area but still violate the orientation convention used by a computational code, leading to incorrect normal directions, inverted local mappings, or invalid finite-element assembly operations.

The `distance` function computes Euclidean edge lengths between pairs of vertices and forms the basis of the geometric quality calculations implemented later in the program. The resulting edge lengths are used to evaluate the simplex diameter $h_K$, the semiperimeter introduced in Equation (21.7.17), and the inradius and circumradius formulas given in Equations (21.7.18) and (21.7.19). Because these quantities depend only on local edge geometry, they provide purely geometric measures of simplex quality independent of any particular numerical discretization method.

The `angle_from_sides` function evaluates interior triangle angles using the law of cosines. By computing all three angles of a triangle and selecting the minimum value, the program evaluates the minimum-angle quality measure introduced in Equation (21.7.21). Small minimum angles are numerically undesirable because they often lead to poor interpolation behavior, ill-conditioned stiffness matrices, and unstable gradient approximations. The implementation therefore uses the minimum angle as one of several complementary indicators of element quality.

The `affine_condition_number_2x2` function evaluates the conditioning of the affine element map associated with the simplex transformation matrix $B_K$ introduced in Equation (21.7.10). The function constructs the local Jacobian matrix from the triangle edge vectors and evaluates its spectral condition number through the eigenvalues of the associated symmetric matrix $B_K^TB_K$. This quantity corresponds directly to the condition-number measure introduced in Equation (21.7.25). Large condition numbers indicate stretched or nearly singular elements, which can amplify numerical errors when gradients are transformed according to Equation (21.7.13). The implementation therefore connects geometric distortion directly to finite-element conditioning behavior.

The `triangle_quality` function provides the central mesh-diagnostic workflow of the program. Given a triangle and the associated vertex coordinates, the function computes the signed area, geometric area, edge lengths, inradius, circumradius, radius-ratio quality, minimum angle, aspect ratio, affine-map condition number, orientation classification, and validity classification. The radius-ratio quality measure follows Equation (21.7.16), while the aspect ratio corresponds to the quantity introduced in Equation (21.7.23). The function also classifies elements as valid, nearly degenerate, or degenerate according to geometric tolerances and conditioning behavior. This combined analysis reflects the broader principle emphasized in Section 21.7.2: mesh quality is simultaneously geometric and numerical.

The helper functions `print_point` and `print_triangle` separate output formatting from geometric computation. These functions present vertex coordinates and triangle diagnostics in a structured format suitable for direct inspection and validation. By reporting signed areas, orientation classifications, minimum angles, aspect ratios, and condition numbers together, the diagnostic output exposes the relationship between geometric distortion and numerical quality.

The `main` function serves to demonstrate mesh orientation and quality analysis using a small two-dimensional triangular mesh. It begins by constructing a set of vertices together with several representative triangles, including well-shaped elements, a highly stretched near-degenerate element, and a deliberately reversed clockwise element. Each triangle is then analyzed using the `triangle_quality` function, after which the resulting diagnostics are printed individually. The program also accumulates global mesh statistics such as total mesh area, the number of valid elements, the number of clockwise elements, and the worst radius-ratio quality observed across the mesh. The resulting output demonstrates how geometric orientation, element distortion, and affine-map conditioning interact to determine whether a mesh is suitable for stable scientific computation.

```rust
// Program 21.7.1: Triangle Mesh Orientation, Area, and Quality Diagnostics
//
// Problem statement:
// Given a two-dimensional triangular mesh stored by vertex coordinates and
// triangle connectivity, compute orientation, signed area, absolute area,
// side lengths, radius-ratio quality, minimum angle, aspect ratio, and the
// condition number of the affine element map. The program follows the mesh
// validity and quality ideas in Section 21.7.1 and Section 21.7.2.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    vertices: [usize; 3],
}

#[derive(Clone, Debug)]
struct TriangleQuality {
    signed_area: f64,
    area: f64,
    edge_lengths: [f64; 3],
    inradius: f64,
    circumradius: f64,
    radius_ratio: f64,
    min_angle_degrees: f64,
    aspect_ratio: f64,
    affine_condition_number: f64,
    orientation: Orientation,
    validity: ElementValidity,
}

#[derive(Clone, Copy, Debug)]
enum Orientation {
    CounterClockwise,
    Clockwise,
    Degenerate,
}

#[derive(Clone, Copy, Debug)]
enum ElementValidity {
    Valid,
    NearlyDegenerate,
    Degenerate,
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn signed_area(a: Point2, b: Point2, c: Point2) -> f64 {
    0.5 * orient2d(a, b, c)
}

fn distance(a: Point2, b: Point2) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn clamp_for_acos(x: f64) -> f64 {
    x.max(-1.0).min(1.0)
}

fn angle_from_sides(adjacent_1: f64, adjacent_2: f64, opposite: f64) -> f64 {
    let denominator = 2.0 * adjacent_1 * adjacent_2;

    if denominator == 0.0 {
        return 0.0;
    }

    let cosine =
        (adjacent_1 * adjacent_1 + adjacent_2 * adjacent_2 - opposite * opposite) / denominator;

    clamp_for_acos(cosine).acos()
}

fn affine_condition_number_2x2(a: Point2, b: Point2, c: Point2) -> f64 {
    let b11 = b.x - a.x;
    let b12 = c.x - a.x;
    let b21 = b.y - a.y;
    let b22 = c.y - a.y;

    let s11 = b11 * b11 + b21 * b21;
    let s12 = b11 * b12 + b21 * b22;
    let s22 = b12 * b12 + b22 * b22;

    let trace = s11 + s22;
    let determinant = s11 * s22 - s12 * s12;

    if determinant <= 0.0 {
        return f64::INFINITY;
    }

    let discriminant = (trace * trace - 4.0 * determinant).max(0.0).sqrt();
    let lambda_max = 0.5 * (trace + discriminant);
    let lambda_min = 0.5 * (trace - discriminant);

    if lambda_min <= 0.0 {
        f64::INFINITY
    } else {
        (lambda_max / lambda_min).sqrt()
    }
}

fn triangle_quality(vertices: &[Point2], triangle: Triangle) -> TriangleQuality {
    let [i0, i1, i2] = triangle.vertices;

    let v0 = vertices[i0];
    let v1 = vertices[i1];
    let v2 = vertices[i2];

    let signed_area_value = signed_area(v0, v1, v2);
    let area = signed_area_value.abs();

    let a = distance(v1, v2);
    let b = distance(v0, v2);
    let c = distance(v0, v1);

    let semiperimeter = 0.5 * (a + b + c);
    let diameter = a.max(b).max(c);

    let inradius = if semiperimeter > 0.0 {
        area / semiperimeter
    } else {
        0.0
    };

    let circumradius = if area > 0.0 {
        a * b * c / (4.0 * area)
    } else {
        f64::INFINITY
    };

    let radius_ratio = if circumradius.is_finite() && circumradius > 0.0 {
        2.0 * inradius / circumradius
    } else {
        0.0
    };

    let angle0 = angle_from_sides(b, c, a);
    let angle1 = angle_from_sides(a, c, b);
    let angle2 = angle_from_sides(a, b, c);

    let min_angle_degrees = angle0.min(angle1).min(angle2).to_degrees();

    let aspect_ratio = if inradius > 0.0 {
        diameter / inradius
    } else {
        f64::INFINITY
    };

    let affine_condition_number = affine_condition_number_2x2(v0, v1, v2);

    let tolerance = 1.0e-12;

    let orientation = if signed_area_value > tolerance {
        Orientation::CounterClockwise
    } else if signed_area_value < -tolerance {
        Orientation::Clockwise
    } else {
        Orientation::Degenerate
    };

    let validity = if area == 0.0 {
        ElementValidity::Degenerate
    } else if area < tolerance || radius_ratio < 1.0e-8 || !affine_condition_number.is_finite() {
        ElementValidity::NearlyDegenerate
    } else {
        ElementValidity::Valid
    };

    TriangleQuality {
        signed_area: signed_area_value,
        area,
        edge_lengths: [a, b, c],
        inradius,
        circumradius,
        radius_ratio,
        min_angle_degrees,
        aspect_ratio,
        affine_condition_number,
        orientation,
        validity,
    }
}

fn print_point(index: usize, p: Point2) {
    println!("v_{:<2} = ({:>10.6}, {:>10.6})", index, p.x, p.y);
}

fn print_triangle(index: usize, triangle: Triangle, quality: &TriangleQuality) {
    println!("Triangle {}", index);
    println!("----------");
    println!(
        "vertices                  = ({}, {}, {})",
        triangle.vertices[0], triangle.vertices[1], triangle.vertices[2]
    );
    println!("signed area               = {:>14.8}", quality.signed_area);
    println!("absolute area             = {:>14.8}", quality.area);
    println!(
        "edge lengths              = [{:>10.6}, {:>10.6}, {:>10.6}]",
        quality.edge_lengths[0], quality.edge_lengths[1], quality.edge_lengths[2]
    );
    println!("inradius                  = {:>14.8}", quality.inradius);
    println!("circumradius              = {:>14.8}", quality.circumradius);
    println!("radius-ratio quality      = {:>14.8}", quality.radius_ratio);
    println!(
        "minimum angle degrees     = {:>14.8}",
        quality.min_angle_degrees
    );
    println!("aspect ratio h_K / rho_K  = {:>14.8}", quality.aspect_ratio);
    println!(
        "condition number kappa(B) = {:>14.8}",
        quality.affine_condition_number
    );
    println!("orientation               = {:?}", quality.orientation);
    println!("validity                  = {:?}", quality.validity);
    println!();
}

fn main() {
    let vertices = vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 { x: 1.0, y: 0.0 },
        Point2 { x: 0.0, y: 1.0 },
        Point2 { x: 1.0, y: 1.0 },
        Point2 { x: 2.0, y: 0.0 },
        Point2 { x: 2.0, y: 0.05 },
        Point2 { x: 3.0, y: 0.0 },
    ];

    let triangles = vec![
        Triangle {
            vertices: [0, 1, 2],
        },
        Triangle {
            vertices: [1, 3, 2],
        },
        Triangle {
            vertices: [4, 6, 5],
        },
        Triangle {
            vertices: [0, 2, 1],
        },
    ];

    println!("Triangle Mesh Orientation and Quality Diagnostics");
    println!("=================================================\n");

    println!("Vertices");
    println!("--------");
    for (i, p) in vertices.iter().enumerate() {
        print_point(i, *p);
    }
    println!();

    println!("Triangle Diagnostics");
    println!("--------------------\n");

    let mut total_area = 0.0;
    let mut valid_count = 0usize;
    let mut clockwise_count = 0usize;
    let mut degenerate_count = 0usize;
    let mut worst_quality = f64::INFINITY;
    let mut worst_triangle = 0usize;

    for (i, triangle) in triangles.iter().enumerate() {
        let quality = triangle_quality(&vertices, *triangle);

        total_area += quality.area;

        if matches!(quality.validity, ElementValidity::Valid) {
            valid_count += 1;
        }

        if matches!(quality.orientation, Orientation::Clockwise) {
            clockwise_count += 1;
        }

        if matches!(
            quality.validity,
            ElementValidity::Degenerate | ElementValidity::NearlyDegenerate
        ) {
            degenerate_count += 1;
        }

        if quality.radius_ratio < worst_quality {
            worst_quality = quality.radius_ratio;
            worst_triangle = i;
        }

        print_triangle(i, *triangle, &quality);
    }

    println!("Mesh Summary");
    println!("------------");
    println!("number of vertices        = {}", vertices.len());
    println!("number of triangles       = {}", triangles.len());
    println!("total absolute area       = {:.8}", total_area);
    println!("valid elements            = {}", valid_count);
    println!("clockwise elements        = {}", clockwise_count);
    println!("degenerate or near-degenerate elements = {}", degenerate_count);
    println!("worst radius-ratio element = {}", worst_triangle);
    println!("worst radius-ratio quality = {:.8}", worst_quality);
}
```

Program 21.7.1 demonstrates how geometric validity and numerical quality diagnostics can be integrated into a unified mesh-analysis framework for simplicial meshes. By combining orientation predicates, affine-map conditioning analysis, radius-ratio measures, aspect ratios, and minimum-angle diagnostics, the implementation reveals how mesh geometry directly influences numerical stability and approximation reliability.

The example illustrates several important classes of mesh behavior. Well-shaped counterclockwise elements exhibit moderate aspect ratios, strong radius-ratio quality, and well-conditioned affine maps. In contrast, highly stretched elements display small minimum angles, poor radius-ratio quality, and large affine condition numbers, indicating potential instability in finite-element gradient transformations and stiffness-matrix assembly. The program also identifies orientation inconsistencies explicitly, demonstrating how inverted elements can be detected through signed-area evaluation before they corrupt downstream numerical operations.

The implementation further illustrates the broader computational philosophy emphasized throughout Section 21.7: robust mesh processing requires the separation of topology, geometry, and numerical diagnostics. Connectivity determines incidence structure, geometric coordinates determine element shape, and quality measures determine computational suitability. This modular organization provides a foundation for more advanced mesh-processing operations such as adaptive refinement, curved-element validation, mesh optimization, point location, barycentric interpolation, and finite-element assembly. The program therefore serves both as a practical mesh-diagnostic tool and as a conceptual introduction to the geometric foundations underlying scientific-computing discretizations.

## 21.7.3. Barycentric Coordinates and Interpolation on Cells

Barycentric coordinates are the primary local coordinate system on simplices. Let,

$$K=\operatorname{conv}\{v_0,v_1,\ldots,v_d\}\tag{21.7.29}$$

be a nondegenerate simplex in $\mathbb{R}^d$. A point $x\in\mathbb{R}^d$ has barycentric coordinates $\lambda_0,\ldots,\lambda_d$ with respect to $K$ if,

$$x=\sum_{i=0}^d \lambda_i v_i,\qquad\sum_{i=0}^d \lambda_i=1 \tag{21.7.30}$$

The point lies inside the closed simplex if and only if:

$$\lambda_i\ge 0,\qquad i=0,\ldots,d \tag{21.7.31}$$

It lies in the interior if:

$$\lambda_i>0,\qquad i=0,\ldots,d \tag{21.7.32}$$

and on the boundary if at least one coordinate is zero while all are nonnegative.

For a triangle $K=(v_0,v_1,v_2)$, the barycentric coordinates can be written using signed areas:

$$
\lambda_0(x)
=
\frac{\operatorname{orient2d}(x,v_1,v_2)}
{\operatorname{orient2d}(v_0,v_1,v_2)}
\tag{21.7.33}
$$

$$
\lambda_1(x)
=
\frac{\operatorname{orient2d}(v_0,x,v_2)}
{\operatorname{orient2d}(v_0,v_1,v_2)}
\tag{21.7.34}
$$

$$
\lambda_2(x)
=
\frac{\operatorname{orient2d}(v_0,v_1,x)}
{\operatorname{orient2d}(v_0,v_1,v_2)}
\tag{21.7.35}
$$

For a tetrahedron $K=(v_0,v_1,v_2,v_3)$, the coordinates are signed volume ratios:

$$
\lambda_0(x)
=
\frac{\operatorname{orient3d}(x,v_1,v_2,v_3)}
{\operatorname{orient3d}(v_0,v_1,v_2,v_3)}
\tag{21.7.36}
$$

$$
\lambda_1(x)
=
\frac{\operatorname{orient3d}(v_0,x,v_2,v_3)}
{\operatorname{orient3d}(v_0,v_1,v_2,v_3)}
\tag{21.7.37}
$$

$$
\lambda_2(x)
=
\frac{\operatorname{orient3d}(v_0,v_1,x,v_3)}
{\operatorname{orient3d}(v_0,v_1,v_2,v_3)}
\tag{21.7.38}
$$

$$
\lambda_3(x)
=
\frac{\operatorname{orient3d}(v_0,v_1,v_2,x)}
{\operatorname{orient3d}(v_0,v_1,v_2,v_3)}
\tag{21.7.39}
$$

For finite element interpolation, the barycentric coordinates are the linear shape functions:

$$N_i(x)=\lambda_i(x),\qquad i=0,\ldots,d \tag{21.7.40}$$

Given nodal values $u_i=u(v_i)$, the linear interpolant is:

$$I_Ku(x) = \sum_{i=0}^d \lambda_i(x)u_i \tag{21.7.41}$$

The partition-of-unity property,

$$\sum_{i=0}^d \lambda_i(x)=1\tag{21.7.42}$$

implies that constants are reproduced exactly:

$$u_i=c\ \text{for all }i\quad\Longrightarrow\quad I_Ku(x)=c \tag{21.7.43}$$

The affine precision property states that every affine function is reproduced exactly. If:

$$u(x)=\alpha+\beta\cdot x \tag{21.7.44}$$

then,

$$I_Ku(x)=u(x)\qquad\text{for all }x\in K \tag{21.7.45}$$

This follows directly from equation (21.7.30). The gradients of barycentric coordinates are constant on an affine simplex. Using the matrix:

$$
B_K
=
\begin{bmatrix}
v_1-v_0 & \cdots & v_d-v_0
\end{bmatrix}
\tag{21.7.46}
$$

one obtains,

$$(\lambda_1,\ldots,\lambda_d)^T = B_K^{-1}(x-v_0) \tag{21.7.47}$$

and

$$\lambda_0 = 1-\sum_{i=1}^d \lambda_i \tag{21.7.48}$$

Therefore,

$$\nabla \lambda_i = B_K^{-T}e_i,\qquad i=1,\ldots,d \tag{21.7.49}$$

and

$$\nabla \lambda_0 = -\sum_{i=1}^d \nabla \lambda_i \tag{21.7.50}$$

These formulas are central in finite element stiffness assembly, because gradients of linear shape functions determine local derivative operators.

On polygonal or polyhedral cells, ordinary simplex barycentric coordinates no longer apply directly. Generalized barycentric coordinates extend the same principles to more general geometries. A set of coordinate functions $\lambda_i(x)$ associated with polygon vertices $v_i$ should satisfy at least:

$$\sum_i \lambda_i(x)=1 \tag{21.7.51}$$

and

$$x=\sum_i \lambda_i(x)v_i \tag{21.7.52}$$

Nonnegativity,

$$\lambda_i(x)\ge 0 \tag{21.7.53}$$

is desirable inside convex cells because it supports stable interpolation and containment reasoning. Recent work on nonnegative moment coordinates and stable mean value coordinates shows that generalized barycentric coordinates remain a technically active area, especially for finite element geometries that go beyond standard triangles and tetrahedra (Dieci, Difonzo and Sukumar, 2024; Fuda and Hormann, 2024).

### Rust Implementation

Following the discussion in Section 21.7.3 on barycentric coordinates, simplex interpolation, and affine finite-element mappings, Program 21.7.2 provides a practical implementation of barycentric-coordinate evaluation, point classification, and linear interpolation on triangular cells. The program demonstrates how barycentric coordinates serve simultaneously as local geometric coordinates, containment predicates, and linear finite-element shape functions. Using signed orientation ratios, the implementation computes barycentric coordinates for arbitrary query points, classifies points relative to the triangle interior and boundary, reconstructs physical coordinates from barycentric weights, and evaluates linear interpolants from nodal data. The program also verifies the partition-of-unity and affine-precision properties central to finite-element interpolation theory. In addition, the implementation computes constant barycentric gradients derived from the affine simplex transformation matrix, thereby connecting local interpolation geometry directly to the derivative operators used in finite-element stiffness assembly. The resulting framework illustrates how barycentric coordinates unify geometry, interpolation, and local differential operators within simplex-based numerical methods.

At the core of the implementation are the `Point2` and `Triangle` structures, which represent two-dimensional geometric coordinates and simplicial connectivity respectively. The `Triangle` structure stores the three physical vertices defining the simplex, while the `BarycentricCoordinates` structure stores the corresponding coordinate weights $\lambda_0,\lambda_1,\lambda_2$. The `PointClassification` enumeration provides a geometric classification of query points as interior, boundary, or exterior points according to the sign conditions introduced in Equations (21.7.31) and (21.7.32). The `Gradient2` structure stores the constant spatial gradients of the barycentric coordinate functions, thereby linking interpolation geometry directly to derivative evaluation.

The `orient2d` function implements the signed orientation predicate used throughout the barycentric-coordinate construction. Given three planar points, the function evaluates the oriented determinant associated with the triangle area. This determinant forms the denominator appearing in Equations (21.7.33)–(21.7.35) and therefore determines both simplex orientation and barycentric normalization. Because barycentric coordinates are fundamentally ratios of signed subareas, the orientation predicate provides the geometric foundation of the entire interpolation framework.

The `barycentric_coordinates` function computes the barycentric coordinate weights associated with a query point relative to a triangle. The implementation follows the signed-area formulas introduced in Equations (21.7.33)–(21.7.35). Each barycentric coordinate is computed as the ratio between the signed area of a subtriangle and the signed area of the full simplex. The function first evaluates the orientation denominator associated with the parent triangle and then computes the three signed subarea ratios corresponding to the coordinate weights. If the denominator is numerically zero, the triangle is degenerate and the function returns `None`. Otherwise, the resulting barycentric coordinates satisfy the partition-of-unity relation introduced in Equation (21.7.42).

The `classify_point` function determines whether a query point lies in the interior, on the boundary, or outside the simplex. The implementation follows the sign conditions described in Equations (21.7.31) and (21.7.32). If all barycentric coordinates are strictly positive, the point is classified as interior. If at least one coordinate is numerically zero while the remaining coordinates are nonnegative, the point lies on the boundary. If any coordinate is negative beyond the numerical tolerance, the point lies outside the simplex. This classification logic illustrates the important geometric interpretation of barycentric coordinates as containment predicates.

The `linear_interpolate` function implements the linear finite-element interpolant introduced in Equation (21.7.41). Given barycentric coordinate weights and nodal values $u_i$, the function evaluates the weighted sum,

$$I_Ku(x)=\sum_{i=0}^{2}\lambda_i(x)u_i$$

Because barycentric coordinates form a partition of unity, the resulting interpolant reproduces constant fields exactly, as described in Equation (21.7.43). More generally, because barycentric coordinates satisfy the affine reconstruction property introduced in Equation (21.7.30), the interpolant reproduces all affine functions exactly, consistent with Equation (21.7.45).

The `barycentric_gradients` function computes the constant gradients of the barycentric coordinate functions over the simplex. The implementation constructs the affine transformation matrix $B_K$ introduced in Equation (21.7.46), computes its inverse transpose, and extracts the gradients according to Equation (21.7.49). The gradient of $\lambda_0$ is then obtained through Equation (21.7.50). Because the simplex mapping is affine, these gradients are spatially constant throughout the element. This property is fundamental in finite-element methods because local stiffness matrices depend directly on products of shape-function gradients.

The `affine_function` function provides a test affine field of the form introduced in Equation (21.7.44). This function is used to verify the affine-precision property of barycentric interpolation. By evaluating the affine field at the triangle vertices and then reconstructing the field through barycentric interpolation, the implementation demonstrates numerically that the interpolated field reproduces the exact affine value at arbitrary query points.

The `reconstructed_point` function verifies the geometric reconstruction property of barycentric coordinates. Using the barycentric weights together with the triangle vertices, the function reconstructs the physical query point according to Equation (21.7.30). The resulting reconstructed coordinates provide a direct numerical confirmation that barycentric coordinates act as exact affine local coordinates on the simplex.

The printing functions separate diagnostic output from geometric computation. Functions such as `print_point`, `print_barycentric`, and `print_gradients` provide structured reporting of geometric coordinates, barycentric weights, interpolation results, and gradient vectors. The diagnostic output explicitly displays the partition-of-unity property, interpolation accuracy, affine-reproduction behavior, and point classification results, thereby making the theoretical properties of barycentric coordinates directly visible in numerical form.

The `main` function serves to demonstrate barycentric-coordinate evaluation and finite-element interpolation on a representative triangular simplex. It begins by constructing a nondegenerate triangle together with several query points representing interior, boundary, and exterior configurations. The program then computes barycentric coordinates, classifies each query point, reconstructs the physical coordinates from the barycentric representation, evaluates linear interpolants from prescribed nodal data, and verifies exact reproduction of both constant and affine functions. Finally, the program computes and prints the constant barycentric gradients associated with the affine simplex map. The resulting output demonstrates how barycentric coordinates simultaneously support geometric containment, interpolation, affine reconstruction, and finite-element derivative evaluation within a unified local-coordinate framework.

```rust
// Program 21.7.2: Barycentric Coordinates, Point Classification,
// and Linear Triangle Interpolation
//
// Problem statement:
// Given a nondegenerate triangle K = conv{v_0, v_1, v_2} and a query point x,
// compute barycentric coordinates using signed-area ratios, classify the point
// as interior, boundary, or exterior, evaluate the linear finite element
// interpolant, and compute constant barycentric gradients. The implementation
// follows the barycentric-coordinate, interpolation, and gradient formulas in
// Section 21.7.3.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    vertices: [Point2; 3],
}

#[derive(Clone, Copy, Debug)]
struct BarycentricCoordinates {
    lambda: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
enum PointClassification {
    Interior,
    Boundary,
    Exterior,
}

#[derive(Clone, Copy, Debug)]
struct Gradient2 {
    dx: f64,
    dy: f64,
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn barycentric_coordinates(
    triangle: Triangle,
    x: Point2,
) -> Option<BarycentricCoordinates> {
    let v0 = triangle.vertices[0];
    let v1 = triangle.vertices[1];
    let v2 = triangle.vertices[2];

    let denominator = orient2d(v0, v1, v2);

    if denominator.abs() <= 1.0e-14 {
        return None;
    }

    let lambda0 = orient2d(x, v1, v2) / denominator;
    let lambda1 = orient2d(v0, x, v2) / denominator;
    let lambda2 = orient2d(v0, v1, x) / denominator;

    Some(BarycentricCoordinates {
        lambda: [lambda0, lambda1, lambda2],
    })
}

fn classify_point(
    bary: BarycentricCoordinates,
    tolerance: f64,
) -> PointClassification {
    let mut has_negative = false;
    let mut has_near_zero = false;

    for &lambda_i in &bary.lambda {
        if lambda_i < -tolerance {
            has_negative = true;
        }

        if lambda_i.abs() <= tolerance {
            has_near_zero = true;
        }
    }

    if has_negative {
        PointClassification::Exterior
    } else if has_near_zero {
        PointClassification::Boundary
    } else {
        PointClassification::Interior
    }
}

fn linear_interpolate(
    bary: BarycentricCoordinates,
    nodal_values: [f64; 3],
) -> f64 {
    bary.lambda[0] * nodal_values[0]
        + bary.lambda[1] * nodal_values[1]
        + bary.lambda[2] * nodal_values[2]
}

fn barycentric_gradients(triangle: Triangle) -> Option<[Gradient2; 3]> {
    let v0 = triangle.vertices[0];
    let v1 = triangle.vertices[1];
    let v2 = triangle.vertices[2];

    let b11 = v1.x - v0.x;
    let b12 = v2.x - v0.x;
    let b21 = v1.y - v0.y;
    let b22 = v2.y - v0.y;

    let determinant = b11 * b22 - b12 * b21;

    if determinant.abs() <= 1.0e-14 {
        return None;
    }

    let inv_t_11 = b22 / determinant;
    let inv_t_12 = -b21 / determinant;
    let inv_t_21 = -b12 / determinant;
    let inv_t_22 = b11 / determinant;

    let grad_lambda1 = Gradient2 {
        dx: inv_t_11,
        dy: inv_t_12,
    };

    let grad_lambda2 = Gradient2 {
        dx: inv_t_21,
        dy: inv_t_22,
    };

    let grad_lambda0 = Gradient2 {
        dx: -grad_lambda1.dx - grad_lambda2.dx,
        dy: -grad_lambda1.dy - grad_lambda2.dy,
    };

    Some([grad_lambda0, grad_lambda1, grad_lambda2])
}

fn affine_function(x: Point2, alpha: f64, beta: Point2) -> f64 {
    alpha + beta.x * x.x + beta.y * x.y
}

fn reconstructed_point(
    bary: BarycentricCoordinates,
    triangle: Triangle,
) -> Point2 {
    let v0 = triangle.vertices[0];
    let v1 = triangle.vertices[1];
    let v2 = triangle.vertices[2];

    Point2 {
        x: bary.lambda[0] * v0.x + bary.lambda[1] * v1.x + bary.lambda[2] * v2.x,
        y: bary.lambda[0] * v0.y + bary.lambda[1] * v1.y + bary.lambda[2] * v2.y,
    }
}

fn print_point(label: &str, p: Point2) {
    println!("{label}({:>10.6}, {:>10.6})", p.x, p.y);
}

fn print_barycentric(label: &str, bary: BarycentricCoordinates) {
    println!("{label}");
    println!("{}", "-".repeat(label.len()));
    println!("lambda_0 = {:>14.10}", bary.lambda[0]);
    println!("lambda_1 = {:>14.10}", bary.lambda[1]);
    println!("lambda_2 = {:>14.10}", bary.lambda[2]);
    println!(
        "sum      = {:>14.10}",
        bary.lambda[0] + bary.lambda[1] + bary.lambda[2]
    );
    println!();
}

fn print_gradients(gradients: [Gradient2; 3]) {
    println!("Barycentric Gradients");
    println!("---------------------");
    for (i, grad) in gradients.iter().enumerate() {
        println!(
            "grad lambda_{} = ({:>12.8}, {:>12.8})",
            i, grad.dx, grad.dy
        );
    }
    println!();
}

fn main() {
    let triangle = Triangle {
        vertices: [
            Point2 { x: 0.0, y: 0.0 },
            Point2 { x: 2.0, y: 0.0 },
            Point2 { x: 0.5, y: 1.5 },
        ],
    };

    let query_points = vec![
        ("interior point", Point2 { x: 0.8, y: 0.4 }),
        ("boundary point", Point2 { x: 1.0, y: 0.0 }),
        ("exterior point", Point2 { x: 2.0, y: 1.0 }),
    ];

    let nodal_values = [1.0, 3.0, 2.0];

    let alpha = 2.0;
    let beta = Point2 { x: -0.5, y: 1.25 };

    println!("Barycentric Coordinates and Linear Triangle Interpolation");
    println!("=========================================================\n");

    println!("Triangle Vertices");
    println!("-----------------");
    for (i, vertex) in triangle.vertices.iter().enumerate() {
        print_point(&format!("v_{} = ", i), *vertex);
    }
    println!();

    println!("Nodal Values");
    println!("------------");
    println!("u(v_0) = {:.6}", nodal_values[0]);
    println!("u(v_1) = {:.6}", nodal_values[1]);
    println!("u(v_2) = {:.6}", nodal_values[2]);
    println!();

    let gradients = barycentric_gradients(triangle)
        .expect("triangle must be nondegenerate to compute gradients");
    print_gradients(gradients);

    for (label, query) in query_points {
        println!("Query: {}", label);
        println!("{}", "-".repeat(7 + label.len()));
        print_point("x = ", query);

        let bary = barycentric_coordinates(triangle, query)
            .expect("triangle must be nondegenerate");

        print_barycentric("Barycentric Coordinates", bary);

        let classification = classify_point(bary, 1.0e-12);
        println!("classification      = {:?}", classification);

        let reconstructed = reconstructed_point(bary, triangle);
        print_point("reconstructed x = ", reconstructed);

        let interpolated = linear_interpolate(bary, nodal_values);
        println!("linear interpolant  = {:>14.10}", interpolated);

        let constant_values = [5.0, 5.0, 5.0];
        let constant_interp = linear_interpolate(bary, constant_values);
        println!("constant test value = {:>14.10}", constant_interp);

        let affine_nodal_values = [
            affine_function(triangle.vertices[0], alpha, beta),
            affine_function(triangle.vertices[1], alpha, beta),
            affine_function(triangle.vertices[2], alpha, beta),
        ];

        let affine_interp = linear_interpolate(bary, affine_nodal_values);
        let affine_exact = affine_function(query, alpha, beta);

        println!("affine interpolant  = {:>14.10}", affine_interp);
        println!("affine exact value  = {:>14.10}", affine_exact);
        println!(
            "affine abs error    = {:>14.6e}",
            (affine_interp - affine_exact).abs()
        );

        println!();
    }
}
```

Program 21.7.2 demonstrates how barycentric coordinates unify geometric representation, containment testing, interpolation, and local differential operators on simplicial meshes. By expressing points as affine combinations of simplex vertices, barycentric coordinates provide a numerically stable and geometrically meaningful local coordinate system for finite-element and interpolation computations.

The example illustrates several of the central theoretical properties discussed in Section 21.7.3. The barycentric coordinates satisfy the partition-of-unity condition exactly, enabling exact reproduction of constant fields. The affine reconstruction property further guarantees exact interpolation of affine functions, which is verified numerically through the affine interpolation tests. At the same time, the sign structure of the barycentric coordinates provides a robust geometric containment criterion that distinguishes interior, boundary, and exterior points.

The implementation also demonstrates the close connection between barycentric geometry and finite-element derivative operators. Because the simplex mapping is affine, the barycentric gradients remain constant throughout the element and can therefore be precomputed efficiently for stiffness assembly and derivative evaluation. This constant-gradient structure is one of the primary computational advantages of simplex-based linear finite elements.

More broadly, the program illustrates the computational philosophy emphasized throughout Chapter 21: local geometric coordinates should simultaneously support robust geometric reasoning and stable numerical approximation. The same principles underlying simplex barycentric coordinates extend naturally to generalized barycentric coordinates on polygonal and polyhedral cells, where nonnegativity, affine precision, and partition-of-unity properties remain central objectives in modern interpolation and mesh-processing research.

## 21.7.4. Point Location in Meshes

Point location is the problem of finding the cell $K\in\mathcal{T}_h$ that contains a query point $x$. In its most basic form, the problem is:

$$\text{find }K\in\mathcal{T}_h\quad\text{such that}\quad x\in K \tag{21.7.54}$$

For interpolation, one then computes local coordinates inside that cell. For particle-to-mesh transfer, point location determines where particle data contribute. For visualization, it identifies which element contains a sampled location. For conservative remapping, point location is part of the process of determining overlap between source and target cells.

The naive method checks every cell $K_1,K_2,\ldots,K_N$. If a point-in-cell test costs $O(1)$ for fixed dimension and fixed cell type, then the full scan costs $O(N)$ per query. This is simple and robust for small meshes, but it is too expensive for large repeated queries.

For a simplicial mesh, a point-in-simplex test can be performed by barycentric coordinates. For a triangle or tetrahedron, compute $\lambda_i(x)$ and classify,

$$x\in K\quad\Longleftrightarrow\quad\lambda_i(x)\ge 0\quad\text{for all }i \tag{21.7.55}$$

A strict interior test is:

$$\lambda_i(x)>0\quad\text{for all }i \tag{21.7.56}$$

A boundary test occurs when:

$$\lambda_j(x)=0\quad\text{for at least one }j\tag{21.7.57}$$

and all remaining coordinates are nonnegative. In floating-point arithmetic, the equality in (21.7.59) must be interpreted through a certified predicate or an explicitly documented tolerance policy.

Walking methods exploit mesh adjacency. Starting from an initial cell $K^{(0)}$, one computes local coordinates of $x$. If all coordinates are nonnegative, the point is found. If some coordinate is negative, the point lies outside the face opposite the corresponding vertex. The algorithm then crosses that face into the neighboring cell. In a simplex, the most negative barycentric coordinate often indicates the face through which to move:

$$j^\ast=\arg\min_j \lambda_j(x) \tag{21.7.58}$$

If,

$$\lambda_{j^\ast}(x)<0 \tag{21.7.59}$$

the walk attempts to move across the face opposite $v_{j^\ast}$. This continues until a containing cell is found or the walk exits the mesh boundary.

Walking methods are often efficient when queries are spatially coherent, for example when tracking particles over time or evaluating values along a curve. If consecutive query points are close, the cell found for the previous query is a good starting point for the next. However, walking is not guaranteed to be globally efficient on arbitrary meshes and may fail or cycle if adjacency is inconsistent or predicates are unreliable. Robust implementations therefore include a fallback, such as a spatial index.

Acceleration structures reduce the number of candidate cells. A common strategy is to store bounding boxes for cells. For a cell (K), its axis-aligned bounding box is:

$$B_K=[\ell_1,u_1]\times\cdots\times[\ell_d,u_d] \tag{21.7.60}$$

where,

$$\ell_m=\min_{x\in K}x_m,\qquad u_m=\max_{x\in K}x_m \tag{21.7.61}$$

A point $x$ can lie in $K$ only if $x\in B_K$. Thus, a grid, $k$-d tree, octree, or bounding volume hierarchy can first return a candidate set,

$$C(x)=\{K:B_K\ \text{may contain }x\} \tag{21.7.62}$$

after which exact point-in-cell tests are applied only to cells in $C(x)$. This two-stage structure,

$$
\text{spatial candidate search}
\quad+\quad
\text{exact local containment test} 
\tag{21.7.63}
$$

is the preferred pattern for large meshes.

For general polygonal, polyhedral, or constructive-solid-geometry cells, containment is more difficult than barycentric testing. A point may need to be classified relative to many faces or Boolean primitives. For a convex polyhedron represented as half-spaces,

$$K=\{x\in\mathbb{R}^d:n_j\cdot x\le \beta_j,\ j=1,\ldots,m\} \tag{21.7.64}$$

containment is equivalent to:

$$n_j\cdot x\le \beta_j\qquad\text{for all }j \tag{21.7.65}$$

For nonconvex or CSG-based cells, traversal strategies and short-circuit logic become important. Modern point-containment algorithms for constructive solid geometry highlight that containment is both a geometric and data-structural problem, especially when unbounded primitives or complex Boolean structures are present (Romano et al., 2025).

## 21.7.5. Implementation Structure and Scientific-Computing Consequences

Mesh algorithms should separate coordinates, topology, geometry, and search. Coordinates store vertex positions. Topology stores incidence and adjacency. Geometry computes measures, orientations, mappings, and quality metrics. Search structures accelerate point location and neighborhood queries. A useful conceptual organization is:

$$
\text{vertices}
\longrightarrow
\text{cells and adjacency}
\longrightarrow
\text{element geometry}
\longrightarrow
\text{spatial index}\\
\longrightarrow
\text{point location and interpolation}
\tag{21.7.66}
$$

This organization prevents the code from confusing a cell’s combinatorial identity with its geometric shape.

For Rust implementations, mesh storage should usually use stable integer indices rather than pointer-rich structures. A triangular mesh may store,

$$K_i=(v_{i0},v_{i1},v_{i2};n_{i0},n_{i1},n_{i2}) \tag{21.7.67}$$

where $v_{ij}$ are vertex indices and $n_{ij}$ are neighbor-cell indices or boundary markers. A tetrahedral mesh analogously stores four vertex indices and four neighbor references. This layout supports contiguous memory, explicit ownership, safe mutation, and efficient traversal.

Point-location functions should return structured classifications:

$$\text{Inside}(K),\qquad\text{Boundary}(K,\Gamma),\qquad\text{Outside} \tag{21.7.68}$$

Here $\Gamma$ may identify the local face, edge, or vertex on which the point lies. This is preferable to returning a boolean because interpolation, remapping, and mesh traversal require different behavior for interior and boundary cases. A point on a shared face may be valid for two adjacent cells, and the code must have a deterministic ownership convention.

A reliable implementation should expose both a fast path and a certified fallback. The fast path may use cached inverse matrices, adjacency walking, bounding boxes, and spatial indexes. The fallback should verify the final classification using robust predicates or carefully localized tolerance logic. Conceptually,

$$\text{fast candidate selection}\longrightarrow\text{local coordinate test}\\ \longrightarrow\text{robust verification}\tag{21.7.69}$$

This pattern is especially important for nearly degenerate cells, boundary queries, and mesh interfaces.

The scientific consequences of mesh geometry are broad. Poor element quality can degrade interpolation, inflate condition numbers, and reduce solver robustness. Incorrect point location can place data in the wrong cell. Inconsistent boundary classification can break conservation during remapping. Invalid orientation can reverse normals and corrupt assembly. Unstable generalized coordinates can introduce interpolation artifacts on polygonal or polyhedral cells. Thus, the mesh layer must be treated as part of the numerical method, not merely as input data.

The sections that follow build directly on this principle. Robust finite-precision geometry is required because mesh algorithms make discrete decisions from numerical predicates. A mesh is useful for scientific computing only when its topology, geometry, search structures, and local coordinate systems work together consistently.

### Rust Implementation

Following the discussion in Sections 21.7.4 and 21.7.5 on mesh point location, spatial candidate search, and exact local containment testing, Program 21.7.3 provides a practical implementation of accelerated point location in triangular meshes using bounding-box filtering and barycentric verification. The program demonstrates the preferred two-stage structure introduced in Equation (21.7.63), where a coarse spatial search first identifies a reduced set of candidate cells and exact geometric predicates are then applied only to those candidates. The implementation combines axis-aligned bounding boxes, simplicial barycentric containment tests, and structured classification logic to determine whether a query point lies inside a cell, on a cell boundary, or outside the mesh entirely. The program also illustrates the implementation philosophy emphasized in Section 21.7.5: mesh algorithms should separate geometry, topology, search structures, and interpolation logic into distinct computational layers. By organizing point location around stable integer indices, explicit mesh connectivity, cached geometric bounds, and exact local verification, the implementation provides a robust foundation for interpolation, particle tracking, remapping, and finite-element evaluation on simplicial meshes.

At the core of the implementation are the `Point2`, `TriangleCell`, and `TriangleMesh` structures, which separate geometric coordinates, mesh topology, and spatial-search information. The `Point2` structure stores planar vertex coordinates, while the `TriangleCell` structure stores the vertex indices and neighbor references associated with each simplicial element. This organization directly reflects the mesh-storage layout described in Equation (21.7.67), where cells are represented through stable integer indices rather than pointer-rich recursive structures. The `TriangleMesh` structure stores the global vertex array, cell connectivity, and precomputed axis-aligned bounding boxes associated with each triangle, thereby separating geometry from acceleration structures and traversal logic.

The `Aabb2` structure represents the axis-aligned bounding box,

$$B_K=[\ell_1,u_1]\times[\ell_2,u_2]$$

introduced in Equation (21.7.60). The bounding-box limits are computed using the minimum and maximum coordinate values associated with the triangle vertices, consistent with Equation (21.7.61). These bounding boxes provide the first stage of the accelerated point-location algorithm because a query point can only lie inside a triangle if it also lies inside the triangle’s bounding box.

The `LocationResult` enumeration implements the structured point-classification design introduced in Equation (21.7.68). Instead of returning only a boolean containment flag, the program distinguishes explicitly between interior, boundary, and exterior classifications. Interior and boundary results also store the containing cell index together with the associated barycentric coordinates. Boundary classifications additionally record the local zero-barycentric entities responsible for the boundary condition. This richer classification structure is essential because interpolation, remapping, and traversal algorithms often require different behavior for interior and boundary points.

The `CandidateReport` structure stores the complete diagnostic information associated with a point-location query. In addition to the final classification result, it stores the candidate cells returned by the bounding-box filtering stage. This separation between candidate generation and exact containment verification reflects the implementation philosophy described in Equation (21.7.69), where fast geometric filtering is followed by certified local verification.

The `Aabb2::from_triangle` function constructs the axis-aligned bounding box associated with a triangular cell. Given the coordinates of the triangle vertices, the function computes the minimum and maximum coordinate values in each spatial direction and stores the resulting interval bounds. This preprocessing stage allows repeated queries to reuse the same cached geometric bounds efficiently.

The `Aabb2::contains_point` function implements the coarse bounding-box inclusion test. The function determines whether a query point lies inside or sufficiently near the axis-aligned bounding box according to a prescribed tolerance policy. Because the bounding box conservatively encloses the triangle, failure of this test guarantees that the query point cannot lie inside the corresponding simplex. This provides the first-stage spatial pruning mechanism of the point-location algorithm.

The `TriangleMesh::new` function constructs the mesh representation and precomputes the bounding boxes associated with all cells. The implementation first validates the mesh connectivity to ensure that all vertex indices are valid and then constructs the bounding-box cache by applying `Aabb2::from_triangle` to every cell. This preprocessing stage converts geometric information into a reusable spatial-search structure suitable for repeated queries.

The `bounding_box_candidates` function implements the candidate-generation stage of the accelerated point-location algorithm. Given a query point, the function scans the bounding boxes and returns the subset of cells whose axis-aligned bounds may contain the point. This candidate set corresponds directly to the set $C(x)$ introduced in Equation (21.7.62). Although the present implementation uses a direct scan of bounding boxes, the same interface could later be accelerated using grids, k-d trees, octrees, or bounding volume hierarchies without altering the exact containment logic.

The `orient2d` function implements the signed orientation predicate underlying the barycentric-coordinate formulas introduced earlier in Equations (21.7.33)–(21.7.35). This predicate forms the geometric foundation of the exact local containment test because barycentric coordinates are fundamentally ratios of signed simplex areas.

The `barycentric_coordinates` function computes the barycentric coordinate representation of a query point relative to a triangle. Using the signed-area formulas introduced in Equations (21.7.33)–(21.7.35), the function evaluates the barycentric weights associated with the query point. If the triangle is degenerate, the function returns `None`; otherwise, the resulting coordinates satisfy the partition-of-unity property introduced in Equation (21.7.42).

The `classify_barycentric` function performs the exact local containment test based on the sign structure of the barycentric coordinates. The implementation follows the simplex-classification conditions described in Equations (21.7.55)–(21.7.57). If all barycentric coordinates are strictly positive, the point is classified as interior. If one or more coordinates are numerically zero while the remaining coordinates are nonnegative, the point is classified as a boundary point. If any coordinate is negative beyond the prescribed tolerance, the point is classified as exterior. This function therefore provides the exact geometric verification stage of the point-location algorithm.

The `locate_point` function combines candidate generation and exact local verification into the full point-location workflow. The function first computes the bounding-box candidate set and then applies barycentric containment testing only to those candidate cells. If an interior or boundary classification is found, the corresponding structured result is returned immediately. Otherwise, the query point is classified as outside the mesh. This two-stage organization directly implements the computational structure described in Equation (21.7.63).

The helper functions `print_mesh_summary`, `print_point`, `print_barycentric`, `print_candidate_list`, and `print_location_report` separate diagnostic output from geometric computation. These functions present mesh connectivity, bounding-box geometry, candidate-cell filtering, barycentric coordinates, and final point classifications in a structured form suitable for validation and debugging. The resulting diagnostic output makes the interaction between spatial acceleration and exact local predicates directly visible.

The `main` function serves to demonstrate accelerated point location on a small triangular mesh. It begins by constructing a structured two-dimensional simplicial mesh together with explicit neighbor references and cached bounding boxes. Several representative query points are then processed, including interior points, a boundary point, and an exterior point lying outside the mesh domain. For each query, the program first computes the bounding-box candidate set and then applies exact barycentric containment testing to determine the final classification. The resulting output demonstrates how coarse spatial pruning dramatically reduces the number of exact geometric tests required for point location while still preserving exact local classification behavior.

```rust
// Program 21.7.3: Triangular-Mesh Point Location with Bounding-Box Candidate
// Search and Barycentric Verification
//
// Problem statement:
// Given a two-dimensional triangular mesh stored using vertex coordinates and
// triangle connectivity, locate query points by first filtering candidate cells
// with axis-aligned bounding boxes and then applying exact local barycentric
// containment tests. The implementation follows the point-location structure
// of Section 21.7.4 and the mesh-storage organization of Section 21.7.5.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct TriangleCell {
    vertices: [usize; 3],
    neighbors: [Option<usize>; 3],
}

#[derive(Clone, Copy, Debug)]
struct Aabb2 {
    lower: Point2,
    upper: Point2,
}

#[derive(Clone, Copy, Debug)]
struct BarycentricCoordinates {
    lambda: [f64; 3],
}

#[derive(Clone, Debug)]
enum LocationResult {
    Inside {
        cell: usize,
        barycentric: BarycentricCoordinates,
    },
    Boundary {
        cell: usize,
        local_entities: Vec<usize>,
        barycentric: BarycentricCoordinates,
    },
    Outside,
}

#[derive(Clone, Debug)]
struct CandidateReport {
    query: Point2,
    candidates: Vec<usize>,
    result: LocationResult,
}

#[derive(Debug)]
struct TriangleMesh {
    vertices: Vec<Point2>,
    cells: Vec<TriangleCell>,
    boxes: Vec<Aabb2>,
}

impl Aabb2 {
    fn from_triangle(vertices: &[Point2], cell: TriangleCell) -> Self {
        let p0 = vertices[cell.vertices[0]];
        let p1 = vertices[cell.vertices[1]];
        let p2 = vertices[cell.vertices[2]];

        let lower = Point2 {
            x: p0.x.min(p1.x).min(p2.x),
            y: p0.y.min(p1.y).min(p2.y),
        };

        let upper = Point2 {
            x: p0.x.max(p1.x).max(p2.x),
            y: p0.y.max(p1.y).max(p2.y),
        };

        Self { lower, upper }
    }

    fn contains_point(&self, p: Point2, tolerance: f64) -> bool {
        p.x >= self.lower.x - tolerance
            && p.x <= self.upper.x + tolerance
            && p.y >= self.lower.y - tolerance
            && p.y <= self.upper.y + tolerance
    }
}

impl TriangleMesh {
    fn new(vertices: Vec<Point2>, cells: Vec<TriangleCell>) -> Self {
        for cell in &cells {
            for &vertex_index in &cell.vertices {
                assert!(
                    vertex_index < vertices.len(),
                    "triangle contains an invalid vertex index"
                );
            }
        }

        let boxes = cells
            .iter()
            .map(|cell| Aabb2::from_triangle(&vertices, *cell))
            .collect();

        Self {
            vertices,
            cells,
            boxes,
        }
    }

    fn locate_point(&self, query: Point2, tolerance: f64) -> CandidateReport {
        let candidates = self.bounding_box_candidates(query, tolerance);

        for &cell_index in &candidates {
            let cell = self.cells[cell_index];

            if let Some(bary) = self.barycentric_in_cell(cell, query) {
                let classification = classify_barycentric(bary, tolerance);

                match classification {
                    LocalClassification::Interior => {
                        return CandidateReport {
                            query,
                            candidates,
                            result: LocationResult::Inside {
                                cell: cell_index,
                                barycentric: bary,
                            },
                        };
                    }
                    LocalClassification::Boundary(local_entities) => {
                        return CandidateReport {
                            query,
                            candidates,
                            result: LocationResult::Boundary {
                                cell: cell_index,
                                local_entities,
                                barycentric: bary,
                            },
                        };
                    }
                    LocalClassification::Exterior => {}
                }
            }
        }

        CandidateReport {
            query,
            candidates,
            result: LocationResult::Outside,
        }
    }

    fn bounding_box_candidates(&self, query: Point2, tolerance: f64) -> Vec<usize> {
        self.boxes
            .iter()
            .enumerate()
            .filter_map(|(cell_index, bbox)| {
                if bbox.contains_point(query, tolerance) {
                    Some(cell_index)
                } else {
                    None
                }
            })
            .collect()
    }

    fn barycentric_in_cell(
        &self,
        cell: TriangleCell,
        query: Point2,
    ) -> Option<BarycentricCoordinates> {
        let v0 = self.vertices[cell.vertices[0]];
        let v1 = self.vertices[cell.vertices[1]];
        let v2 = self.vertices[cell.vertices[2]];

        barycentric_coordinates(v0, v1, v2, query)
    }

    fn print_mesh_summary(&self) {
        println!("Mesh Summary");
        println!("------------");
        println!("number of vertices  = {}", self.vertices.len());
        println!("number of cells     = {}", self.cells.len());
        println!();

        println!("Cells");
        println!("-----");
        for (i, cell) in self.cells.iter().enumerate() {
            println!(
                "K_{:<2} vertices = ({}, {}, {}), neighbors = ({:?}, {:?}, {:?})",
                i,
                cell.vertices[0],
                cell.vertices[1],
                cell.vertices[2],
                cell.neighbors[0],
                cell.neighbors[1],
                cell.neighbors[2]
            );
        }
        println!();

        println!("Cell Bounding Boxes");
        println!("-------------------");
        for (i, bbox) in self.boxes.iter().enumerate() {
            println!(
                "B_{:<2} = [{:>7.3}, {:>7.3}] x [{:>7.3}, {:>7.3}]",
                i, bbox.lower.x, bbox.upper.x, bbox.lower.y, bbox.upper.y
            );
        }
        println!();
    }
}

#[derive(Clone, Debug)]
enum LocalClassification {
    Interior,
    Boundary(Vec<usize>),
    Exterior,
}

fn orient2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn barycentric_coordinates(
    v0: Point2,
    v1: Point2,
    v2: Point2,
    x: Point2,
) -> Option<BarycentricCoordinates> {
    let denominator = orient2d(v0, v1, v2);

    if denominator.abs() <= 1.0e-14 {
        return None;
    }

    let lambda0 = orient2d(x, v1, v2) / denominator;
    let lambda1 = orient2d(v0, x, v2) / denominator;
    let lambda2 = orient2d(v0, v1, x) / denominator;

    Some(BarycentricCoordinates {
        lambda: [lambda0, lambda1, lambda2],
    })
}

fn classify_barycentric(
    bary: BarycentricCoordinates,
    tolerance: f64,
) -> LocalClassification {
    let mut boundary_entities = Vec::new();

    for (i, &lambda_i) in bary.lambda.iter().enumerate() {
        if lambda_i < -tolerance {
            return LocalClassification::Exterior;
        }

        if lambda_i.abs() <= tolerance {
            boundary_entities.push(i);
        }
    }

    if boundary_entities.is_empty() {
        LocalClassification::Interior
    } else {
        LocalClassification::Boundary(boundary_entities)
    }
}

fn print_point(label: &str, p: Point2) {
    println!("{label}({:>8.4}, {:>8.4})", p.x, p.y);
}

fn print_barycentric(bary: BarycentricCoordinates) {
    println!(
        "barycentric = [{:>10.6}, {:>10.6}, {:>10.6}], sum = {:>10.6}",
        bary.lambda[0],
        bary.lambda[1],
        bary.lambda[2],
        bary.lambda[0] + bary.lambda[1] + bary.lambda[2]
    );
}

fn print_candidate_list(candidates: &[usize]) {
    print!("candidate cells from bounding boxes = [");
    for (i, cell_index) in candidates.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", cell_index);
    }
    println!("]");
}

fn print_location_report(report: &CandidateReport) {
    print_point("query x = ", report.query);
    print_candidate_list(&report.candidates);

    match &report.result {
        LocationResult::Inside { cell, barycentric } => {
            println!("classification = Inside(K_{})", cell);
            print_barycentric(*barycentric);
        }
        LocationResult::Boundary {
            cell,
            local_entities,
            barycentric,
        } => {
            println!(
                "classification = Boundary(K_{}, local zero-lambda indices {:?})",
                cell, local_entities
            );
            print_barycentric(*barycentric);
        }
        LocationResult::Outside => {
            println!("classification = Outside");
        }
    }

    println!();
}

fn main() {
    let vertices = vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 { x: 1.0, y: 0.0 },
        Point2 { x: 2.0, y: 0.0 },
        Point2 { x: 0.0, y: 1.0 },
        Point2 { x: 1.0, y: 1.0 },
        Point2 { x: 2.0, y: 1.0 },
    ];

    let cells = vec![
        TriangleCell {
            vertices: [0, 1, 3],
            neighbors: [Some(1), None, None],
        },
        TriangleCell {
            vertices: [1, 4, 3],
            neighbors: [None, Some(0), Some(2)],
        },
        TriangleCell {
            vertices: [1, 2, 4],
            neighbors: [Some(3), None, Some(1)],
        },
        TriangleCell {
            vertices: [2, 5, 4],
            neighbors: [None, Some(2), None],
        },
    ];

    let mesh = TriangleMesh::new(vertices, cells);

    let queries = vec![
        Point2 { x: 0.25, y: 0.25 },
        Point2 { x: 1.25, y: 0.25 },
        Point2 { x: 1.00, y: 0.50 },
        Point2 { x: 1.75, y: 0.75 },
        Point2 { x: 2.20, y: 0.50 },
    ];

    let tolerance = 1.0e-12;

    println!("Triangular-Mesh Point Location with Bounding-Box Candidate Search");
    println!("=================================================================\n");

    mesh.print_mesh_summary();

    println!("Point-Location Queries");
    println!("----------------------\n");

    for query in queries {
        let report = mesh.locate_point(query, tolerance);
        print_location_report(&report);
    }
}
```

Program 21.7.3 demonstrates the preferred computational structure for robust point location in simplicial meshes: coarse spatial candidate generation followed by exact local geometric verification. By combining bounding-box filtering with barycentric containment testing, the implementation reduces unnecessary geometric work while preserving reliable final classification.

The example illustrates several important aspects of mesh point location. Interior points are identified through strictly positive barycentric coordinates, boundary points are detected through zero-coordinate conditions, and exterior points are rejected either during bounding-box filtering or during exact simplex classification. The boundary example further demonstrates why mesh point-location algorithms should return structured classifications rather than simple boolean results, since interpolation and remapping algorithms often require deterministic handling of shared mesh interfaces.

The implementation also highlights the broader organizational principles emphasized in Section 21.7.5. Geometry, topology, search structures, and local predicates are treated as distinct computational layers. Vertex coordinates define geometry, cell connectivity defines topology, bounding boxes provide acceleration structures, and barycentric predicates provide exact local verification. This layered organization improves robustness, extensibility, and computational clarity.

More broadly, the program illustrates the central scientific-computing principle that mesh search structures should accelerate geometric reasoning rather than replace exact geometry itself. Bounding boxes generate candidate cells efficiently, but final containment decisions remain the responsibility of exact barycentric predicates. This separation between fast candidate selection and exact local verification provides the foundation for robust interpolation, particle tracking, conservative remapping, finite-element evaluation, and mesh-based scientific computation on large geometric domains.

# 21.8. Robustness in Finite-Precision Geometry

Robustness is the culminating issue in computational geometry for scientific computing. The preceding sections have described orientation predicates, segment intersections, polygon containment, convex hulls, Delaunay triangulations, spatial search, and point location. All of these algorithms eventually make discrete decisions from numerical quantities. A determinant is positive, negative, or zero. A point is inside, outside, or on the boundary. Two segments intersect or they do not. A tetrahedron is valid or inverted. A face belongs to a cavity boundary or it does not. These decisions are combinatorial, and a wrong decision can change the topology of the computed object. This is why geometric algorithms fail differently from many smooth numerical algorithms: the error is not merely a small perturbation of a scalar result, but a possible change in connectivity, containment, orientation, or conservation structure. Modern work on floating-point filters, exact mesh arrangements, exact mesh constructive solid geometry, and robust intersection algorithms confirms that finite-precision geometry must be treated as a central part of numerical reliability, not as a secondary implementation detail (Bartels, Fisikopoulos and Weiser, 2023; Guo and Fu, 2024; Lévy, 2025; Chen, Ullrich and Panetta, 2026; López and Hernández, 2024).

## 21.8.1. Discrete Failure Modes in Geometric Algorithms

Let $\Delta$ denote an exact geometric quantity whose sign determines a predicate. In many cases, $\Delta$ is a determinant, such as:

$$\Delta=\operatorname{orient2d}(a,b,c) \tag{21.8.1}$$

$$\Delta=\operatorname{orient3d}(a,b,c,d)\tag{21.8.2}$$

$$\Delta=\operatorname{incircle}(a,b,c,d) \tag{21.8.3}$$

or

$$\Delta=\operatorname{insphere}(a,b,c,d,e) \tag{21.8.4}$$

The mathematical predicate is:

$$\operatorname{sign}(\Delta)\in \{-1,0,+1\} \tag{21.8.5}$$

In floating-point arithmetic, the computed value is not $\Delta$, but an approximation $\widehat{\Delta}$. The predicate is correct only if:

$$
\operatorname{sign}(\widehat{\Delta})
=
\operatorname{sign}(\Delta)
\tag{21.8.6}
$$

If this equality fails, the algorithm branches incorrectly. The difficulty is greatest near the degenerate set $\Delta=0$. For the two-dimensional orientation predicate, this means collinearity. For the three-dimensional orientation predicate, it means coplanarity. For incircle and insphere predicates, it means cocircularity and cosphericity. These are not artificial cases. Meshes generated from structured grids, symmetric geometries, CAD models, sampled surfaces, and adaptively refined domains frequently contain exactly or nearly degenerate configurations. Even when the input is mathematically nondegenerate, the floating-point representation may move it close to a degenerate configuration.

A key feature of geometric algorithms is that wrong signs create discrete errors. If a point $c$ is wrongly classified relative to the directed line through $a,b$, then a segment-intersection test may miss a crossing. If a tetrahedron has an orientation sign evaluated incorrectly, an element may be accepted as valid when it is inverted. If an incircle sign is wrong, a Delaunay algorithm may flip the wrong edge or fail to flip a necessary edge. If a point on a shared mesh face is classified differently by two neighboring cells, a point-location algorithm may become nondeterministic.

These failures are topological. For example, suppose two adjacent triangular cells share an edge, and a query point $x$ lies close to that edge. If one cell evaluates,

$$\lambda_j(x)>0\tag{21.8.7}$$

while the neighboring cell evaluates the corresponding boundary relation as:

$$\lambda_j(x)<0 \tag{21.8.8}$$

then the point may be classified as inside neither cell or inside both cells, depending on the local conventions. In interpolation, this changes the chosen element. In remapping, it changes the transfer region. In mesh traversal, it may cause a walk to exit through the wrong face or cycle.

Similarly, in polygon or polytope clipping, an edge endpoint may be classified using:

$$\psi(x)=n\cdot x-\beta \tag{21.8.9}$$

The intended classification is:

$$\psi(x)<0,\qquad \psi(x)=0,\qquad \psi(x)>0 \tag{21.8.10}$$

If two related tests classify the same endpoint inconsistently, the resulting clipped polygon may contain a missing vertex, a duplicate edge, or a reversed boundary segment. In finite-volume methods, such a local geometric inconsistency may become a conservation error because area or volume fractions are computed from the clipped geometry. This is why robust polytope intersection and spherical intersection algorithms matter directly for scientific computing, especially in conservative remapping, cut-cell methods, and volume-of-fluid calculations (Chen, Ullrich and Panetta, 2026; López and Hernández, 2024).

## 21.8.2. Tolerances, Filters, and Exact Predicate Evaluation

The simplest way to handle near-zero geometric quantities is to introduce a tolerance. A typical rule is:

$$
|\widehat{\Delta}| \le \varepsilon
\quad\Longrightarrow\quad
\widehat{\Delta}\ \text{is treated as zero}
\tag{21.8.11}
$$

This approach is common in engineering codes because it is easy to implement and often works for well-scaled data. However, it is not a complete robustness strategy. A single global tolerance may be too large for small features and too small for large coordinates. A local tolerance may solve one classification problem while creating inconsistency elsewhere. Most importantly, tolerance decisions are not automatically coordinated across related predicates.

A more reliable approach is to use an error bound. Suppose floating-point evaluation produces $\widehat{\Delta}$, and suppose a computable bound $E$ satisfies:

$$|\widehat{\Delta}-\Delta|\le E \tag{21.8.12}$$

If

$$
|\widehat{\Delta}| > E 
\tag{21.8.13}
$$

then the exact value $\Delta$ has the same sign as the computed value $\widehat{\Delta}$. Indeed, if $\widehat{\Delta}>E$, then:

$$\Delta\ge \widehat{\Delta}-E>0 \tag{21.8.14}$$

If $\widehat{\Delta}<-E$, then

$$\Delta\le \widehat{\Delta}+E<0 \tag{21.8.15}$$

Therefore,

$$|\widehat{\Delta}|>E\quad\Longrightarrow\quad\operatorname{sign}(\Delta)=\operatorname{sign}(\widehat{\Delta}) \tag{21.8.16}$$

This is the mathematical basis of a floating-point filter.

A filtered predicate first evaluates the determinant using fast floating-point arithmetic. If condition (21.8.14) holds, the sign is certified and returned immediately. If not, the filter reports that the sign cannot be certified at the current precision. The predicate then falls back to a more accurate method:

$$\text{floating-point evaluation}\longrightarrow\text{certification test}\\ \longrightarrow\text{exact or higher-precision fallback} \tag{21.8.17}$$

The important point is that most ordinary inputs are decided quickly, while difficult near-degenerate inputs receive additional arithmetic only when needed.

Exact fallback may be implemented using expansion arithmetic, rational arithmetic, integer arithmetic after suitable input scaling, arbitrary precision, or symbolic methods, depending on the input representation and required guarantees. The exact method must preserve the mathematical sign of the predicate:

$$\operatorname{sign}_{\mathrm{exact}}(\Delta) = \operatorname{sign}(\Delta)\tag{21.8.18}$$

This is more expensive than ordinary floating-point arithmetic, but it is invoked only when the filter cannot certify the sign. Modern robust predicate work is built around precisely this compromise: keep the common case fast, but make the difficult case correct (Bartels, Fisikopoulos and Weiser, 2023).

Tolerance-based logic can still be useful, but it should be understood as a modeling or engineering policy rather than as an exactness guarantee. A tolerance may intentionally merge features smaller than a prescribed scale, or it may stabilize noisy measured data. In that setting, the tolerance defines a modified geometric problem. By contrast, filtered exact predicates aim to solve the original geometric decision problem as specified by the input coordinates. These two goals are different and should not be confused.

A useful hierarchy of predicate strategies is therefore,

$$\text{naive floating point}\quad<\quad\text{tolerance-based classification}\\ \quad<\quad\text{filtered exact predicates}\quad<\quad\text{fully exact constructions}\tag{21.8.19}$$

The ordering is not merely about accuracy. It is about the strength of the consistency guarantee. Naive floating-point predicates are fast but may branch incorrectly. Tolerances reduce some instability but can introduce inconsistent classifications. Filtered exact predicates certify signs. Fully exact constructions also ensure that newly constructed geometric objects, such as intersection points or split vertices, are represented consistently enough for later predicates and topology updates.

### Introductory Paragraph Before the Code Block

Following the discussion in Sections 21.8.1 and 21.8.2 on floating-point failure modes, tolerance policies, and filtered geometric predicates, Program 21.8.1 provides a practical implementation of a robust orientation-predicate kernel with explicit classification logic. The program demonstrates how geometric predicates should be treated not merely as raw floating-point determinant evaluations, but as certified computational decisions accompanied by numerical diagnostics and uncertainty handling. Using naive sign tests, tolerance-based classification, and filtered error-bound certification, the implementation evaluates the orientation predicate under well-conditioned, degenerate, and nearly degenerate geometric configurations. The program also illustrates the broader robustness philosophy emphasized throughout Section 21.8: geometric kernels should separate numerical evaluation, predicate certification, and final classification policy into distinct computational stages. By returning explicit sign classifications rather than undocumented boolean results, the implementation provides a safer and more extensible foundation for mesh processing, intersection testing, point classification, and scientific geometric computation.

### Explanatory Paragraphs After the Introductory Paragraph

At the core of the implementation are the `Point2`, `Sign`, and `OrientationReport` structures, which separate geometric coordinates, predicate classifications, and numerical diagnostics. The `Point2` structure stores planar coordinates, while the `Sign` enumeration provides an explicit predicate classification system with the states `Positive`, `Negative`, `Zero`, and `Uncertain`. This explicit-sign structure reflects the robust-classification philosophy described in Equations (21.8.35)–(21.8.38), where predicates should return structured classifications rather than undocumented boolean values. The `OrientationReport` structure stores the determinant value, multiple sign interpretations, and the associated certification bound, thereby separating numerical computation from interpretation policy.

The `orient2d_value` function evaluates the raw two-dimensional orientation determinant associated with the predicate

\[\
\\operatorname{orient2d}(a,b,c).\
\]

This determinant corresponds to the signed-area formulation introduced earlier in Equation (21.7.5). Positive values indicate counterclockwise orientation, negative values indicate clockwise orientation, and zero indicates collinearity. Although the function itself performs only a direct floating-point evaluation, the program demonstrates that raw determinant values alone are insufficient for robust geometric classification near degeneracies.

The `sign_from_value` function implements the simplest possible floating-point predicate interpretation by returning the sign of the determinant directly. This naive strategy corresponds to the unfiltered floating-point evaluation discussed in Section 21.8.1. While this approach is computationally inexpensive, it provides no robustness guarantees near degenerate geometric configurations where cancellation and roundoff may corrupt the sign.

The `tolerance_sign` function implements a tolerance-based predicate policy. Instead of testing only whether the determinant is exactly positive or negative, the function introduces a numerical tolerance region around zero. Determinants with magnitude below the tolerance are classified as zero. This approach reflects the tolerance-based classification policy discussed in Equations (21.8.11)–(21.8.13). Although tolerance methods can improve practical stability, the section emphasizes that tolerance selection itself becomes part of the algorithmic specification and may lead to inconsistent classifications if not carefully documented.

The `filtered_orient2d` function implements the filtered-predicate strategy introduced in Equations (21.8.14)–(21.8.17). The function first computes the floating-point determinant and then constructs a certification bound proportional to machine epsilon and the magnitudes of the multiplicative terms appearing in the determinant expansion. If the determinant magnitude exceeds this bound, the sign is certified as reliable. Otherwise, the predicate returns the explicit classification `Uncertain`. This structure reflects the central robustness principle of filtered predicates: floating-point evaluation is accepted only when the sign can be mathematically certified.

The `robust_orientation_report` function combines naive evaluation, tolerance-based interpretation, and filtered certification into a unified diagnostic structure. By reporting all three interpretations simultaneously, the function illustrates how different robustness policies may classify the same near-degenerate configuration differently. This diagnostic structure is especially useful for debugging geometric kernels and understanding how tolerance policies interact with certified filtering methods.

The `point_on_segment` function demonstrates how robust orientation predicates combine with additional geometric constraints to support higher-level geometric reasoning. The function first verifies collinearity using the orientation predicate and then performs interval comparisons to determine whether the point lies within the segment bounds. This two-stage structure reflects the broader computational principle emphasized in Equation (21.8.42): coarse geometric classification should be followed by exact local verification.

The `PointLineClassification` enumeration provides a structured interpretation of orientation results relative to an oriented line. Instead of exposing determinant signs directly to higher-level algorithms, the program converts predicate results into geometric semantic states such as `Left`, `Right`, `OnLine`, and `Uncertain`. This explicit-classification philosophy improves robustness because downstream algorithms operate on documented geometric states rather than implicit floating-point assumptions.

The `classify_point_against_oriented_line` function demonstrates how filtered predicates should be integrated into geometric decision pipelines. The function first attempts to use the certified filtered sign returned by `filtered_orient2d`. If the filtered predicate returns `Uncertain`, the implementation falls back to the tolerance-based policy. This layered structure reflects the robustness pipeline discussed in Equation (21.8.44), where certified filtering is preferred and uncertain cases may be delegated to fallback logic or higher-precision evaluation.

The helper functions `print_point` and `print_orientation_case` separate diagnostic reporting from geometric computation. These functions display coordinates, determinant values, certification bounds, predicate classifications, and geometric interpretations in a structured format suitable for numerical validation. The resulting output makes the interaction between floating-point roundoff, tolerance policies, and certified filtering directly visible.

The `main` function serves to demonstrate robust orientation classification under a range of geometric configurations. It begins by defining several representative test cases, including well-conditioned counterclockwise and clockwise triangles, exactly collinear points, nearly collinear configurations, and large-coordinate near-degenerate cases prone to cancellation effects. For each configuration, the program evaluates the determinant, computes certification bounds, applies multiple predicate-classification strategies, and reports the resulting geometric interpretation. The output therefore demonstrates how robust predicate kernels behave across both stable and numerically challenging geometric regimes.

### Concluding Remarks After the Code Block

```rust
// Program 21.8.1: Robust Orientation Predicate Kernel with Explicit Classification
//
// Problem statement:
// Implement a small robust-geometry predicate kernel for the two-dimensional
// orientation determinant. The program compares naive floating-point signs,
// tolerance-based signs, and a filtered sign test with an error-bound
// certification step. Predicate results are returned as explicit enums,
// following the robustness and implementation principles of Section 21.8.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sign {
    Positive,
    Negative,
    Zero,
    Uncertain,
}

#[derive(Clone, Copy, Debug)]
struct OrientationReport {
    determinant: f64,
    naive_sign: Sign,
    tolerance_sign: Sign,
    filtered_sign: Sign,
    error_bound: f64,
}

fn sign_from_value(value: f64) -> Sign {
    if value > 0.0 {
        Sign::Positive
    } else if value < 0.0 {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

fn tolerance_sign(value: f64, tolerance: f64) -> Sign {
    if value > tolerance {
        Sign::Positive
    } else if value < -tolerance {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

fn filtered_orient2d(a: Point2, b: Point2, c: Point2) -> (Sign, f64, f64) {
    let acx = a.x - c.x;
    let bcx = b.x - c.x;
    let acy = a.y - c.y;
    let bcy = b.y - c.y;

    let term1 = acx * bcy;
    let term2 = acy * bcx;
    let determinant = term1 - term2;

    let machine_epsilon = f64::EPSILON;

    let error_bound = 8.0 * machine_epsilon * (term1.abs() + term2.abs());

    let sign = if determinant > error_bound {
        Sign::Positive
    } else if determinant < -error_bound {
        Sign::Negative
    } else if determinant == 0.0 {
        Sign::Zero
    } else {
        Sign::Uncertain
    };

    (sign, determinant, error_bound)
}

fn robust_orientation_report(
    a: Point2,
    b: Point2,
    c: Point2,
    tolerance: f64,
) -> OrientationReport {
    let (filtered_sign, determinant, error_bound) = filtered_orient2d(a, b, c);

    OrientationReport {
        determinant,
        naive_sign: sign_from_value(determinant),
        tolerance_sign: tolerance_sign(determinant, tolerance),
        filtered_sign,
        error_bound,
    }
}

fn point_on_segment(a: Point2, b: Point2, p: Point2, tolerance: f64) -> bool {
    let report = robust_orientation_report(a, b, p, tolerance);

    if !matches!(report.tolerance_sign, Sign::Zero) {
        return false;
    }

    let min_x = a.x.min(b.x) - tolerance;
    let max_x = a.x.max(b.x) + tolerance;
    let min_y = a.y.min(b.y) - tolerance;
    let max_y = a.y.max(b.y) + tolerance;

    p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y
}

#[derive(Clone, Copy, Debug)]
enum PointLineClassification {
    Left,
    Right,
    OnLine,
    Uncertain,
}

fn classify_point_against_oriented_line(
    a: Point2,
    b: Point2,
    p: Point2,
    tolerance: f64,
) -> PointLineClassification {
    let report = robust_orientation_report(a, b, p, tolerance);

    match report.filtered_sign {
        Sign::Positive => PointLineClassification::Left,
        Sign::Negative => PointLineClassification::Right,
        Sign::Zero => PointLineClassification::OnLine,
        Sign::Uncertain => match report.tolerance_sign {
            Sign::Positive => PointLineClassification::Left,
            Sign::Negative => PointLineClassification::Right,
            Sign::Zero => PointLineClassification::OnLine,
            Sign::Uncertain => PointLineClassification::Uncertain,
        },
    }
}

fn print_point(label: &str, p: Point2) {
    println!("{label}({:>22.15e}, {:>22.15e})", p.x, p.y);
}

fn print_orientation_case(
    label: &str,
    a: Point2,
    b: Point2,
    c: Point2,
    tolerance: f64,
) {
    let report = robust_orientation_report(a, b, c, tolerance);
    let classification = classify_point_against_oriented_line(a, b, c, tolerance);

    println!("{label}");
    println!("{}", "-".repeat(label.len()));
    print_point("a = ", a);
    print_point("b = ", b);
    print_point("c = ", c);
    println!("determinant       = {:>22.15e}", report.determinant);
    println!("error bound       = {:>22.15e}", report.error_bound);
    println!("naive sign        = {:?}", report.naive_sign);
    println!("tolerance sign    = {:?}", report.tolerance_sign);
    println!("filtered sign     = {:?}", report.filtered_sign);
    println!("line classification = {:?}", classification);
    println!(
        "point on segment  = {}",
        point_on_segment(a, b, c, tolerance)
    );
    println!();
}

fn main() {
    let tolerance = 1.0e-12;

    let cases = vec![
        (
            "Counterclockwise triangle",
            Point2 { x: 0.0, y: 0.0 },
            Point2 { x: 1.0, y: 0.0 },
            Point2 { x: 0.0, y: 1.0 },
        ),
        (
            "Clockwise triangle",
            Point2 { x: 0.0, y: 0.0 },
            Point2 { x: 0.0, y: 1.0 },
            Point2 { x: 1.0, y: 0.0 },
        ),
        (
            "Exactly collinear point on segment",
            Point2 { x: 0.0, y: 0.0 },
            Point2 { x: 2.0, y: 2.0 },
            Point2 { x: 1.0, y: 1.0 },
        ),
        (
            "Nearly collinear point",
            Point2 { x: 0.0, y: 0.0 },
            Point2 { x: 1.0, y: 1.0 },
            Point2 {
                x: 2.0,
                y: 2.0 + 1.0e-14,
            },
        ),
        (
            "Large-coordinate near-degeneracy",
            Point2 {
                x: 1.0e9,
                y: 1.0e9,
            },
            Point2 {
                x: 1.0e9 + 1.0,
                y: 1.0e9 + 1.0,
            },
            Point2 {
                x: 1.0e9 + 2.0,
                y: 1.0e9 + 2.0 + 1.0e-7,
            },
        ),
    ];

    println!("Robust Orientation Predicate Kernel");
    println!("===================================\n");

    println!("Predicate Policy");
    println!("----------------");
    println!("tolerance = {:>12.6e}", tolerance);
    println!(
        "filtered signs are certified when |determinant| is larger than the computed error bound"
    );
    println!();

    for (label, a, b, c) in cases {
        print_orientation_case(label, a, b, c, tolerance);
    }

    println!("Summary");
    println!("-------");
    println!("The predicate layer returns explicit classifications instead of raw booleans.");
    println!("Uncertain filtered cases can be routed to a higher-precision or exact fallback.");
}
```

Program 21.8.1 demonstrates how robust geometric predicates should be organized as certified computational decision systems rather than simple floating-point sign tests. By separating raw determinant evaluation, tolerance-based interpretation, filtered certification, and geometric classification into distinct computational stages, the implementation provides a safer and more transparent foundation for geometric computation.

The example illustrates several important robustness behaviors. Well-conditioned geometric configurations produce determinant magnitudes far larger than the certification bounds, allowing filtered predicates to certify the sign immediately. Exactly collinear cases produce determinant values consistent with both tolerance-based and certified-zero classifications. Nearly degenerate configurations, however, demonstrate how naive floating-point signs, tolerance policies, and certified filtering may disagree. These cases illustrate why robust geometric kernels must expose uncertainty explicitly instead of silently returning unreliable classifications.

The implementation also highlights the broader computational philosophy emphasized throughout Section 21.8. Predicate kernels should produce structured geometric classifications that can be composed safely into higher-level algorithms such as mesh validation, segment intersection, polygon clipping, point location, and topological reconstruction. The use of explicit enums, certified filtering, and diagnostic reporting improves both numerical robustness and algorithmic clarity.

More broadly, the program demonstrates the essential role of robust predicates in scientific geometric computation. Spatial data structures, mesh-processing pipelines, interpolation methods, and geometric search algorithms all ultimately depend on reliable local geometric decisions. Filtered predicates provide a principled compromise between exact arithmetic and raw floating-point computation by allowing fast evaluation in well-conditioned cases while exposing uncertain configurations for additional processing. This layered strategy forms the foundation of reliable modern computational geometry kernels.

## 21.8.3. Degeneracy, Symbolic Perturbation, and Consistent Classification

Some inputs are exactly degenerate. Three points may be collinear:

$$\operatorname{orient2d}(a,b,c)=0 \tag{21.8.20}$$

Four points may be cocircular:

$$\operatorname{incircle}(a,b,c,d)=0 \tag{21.8.21}$$

Four points in three dimensions may be coplanar:

$$\operatorname{orient3d}(a,b,c,d)=0 \tag{21.8.22}$$

Five points may be cospherical:

$$\operatorname{insphere}(a,b,c,d,e)=0 \tag{21.8.23}$$

Exact arithmetic can detect these equalities, but detecting degeneracy is not the same as deciding what the algorithm should do with it. The algorithm also needs a degeneracy policy.

One approach is explicit case handling. For segment intersection, the algorithm separately treats proper crossing, endpoint contact, and collinear overlap. For point-in-polygon tests, it first checks whether the query point lies on an edge before applying ray crossing or winding logic. For Delaunay triangulation, cocircular points require a tie-breaking rule if a unique triangulation is desired. These explicit cases are mathematically transparent but can become numerous in high-dimensional or arrangement-level algorithms.

Another approach is symbolic perturbation. The idea is to imagine that the input points have been perturbed by infinitesimal quantities so that degeneracies disappear, while the perturbation is applied consistently. Formally, one may replace a point $p_i$ by:

$$p_i(\epsilon)=p_i+\epsilon q_i+\epsilon^2 r_i+\cdots \tag{21.8.24}$$

where $\epsilon>0$ is infinitesimal and the auxiliary perturbation directions are chosen according to a deterministic rule. A predicate then becomes a formal expression:

$$\Delta(\epsilon) = \Delta_0+\epsilon\Delta_1+\epsilon^2\Delta_2+\cdots \tag{21.8.25}$$

If $\Delta_0=0$, the sign is determined by the first nonzero coefficient:

$$\operatorname{sign}(\Delta(\epsilon)) = \operatorname{sign}(\Delta_k),\qquad k=\min\{j:\Delta_j\ne 0\}\tag{21.8.26}$$

This gives a consistent way to break ties without choosing an arbitrary floating-point tolerance.

Symbolic perturbation is especially useful for algorithms that are simpler under a general-position assumption. A Delaunay triangulation algorithm, for example, may assume that no four points are cocircular in two dimensions and no five points are cospherical in three dimensions. A symbolic perturbation can simulate such a general-position input while preserving a deterministic relationship to the original data. The resulting triangulation is one of the valid choices compatible with the degenerate input.

However, symbolic perturbation must be used carefully in scientific computing. If a degeneracy has physical meaning, such as a boundary point lying exactly on an interface or a mesh vertex shared by multiple material regions, perturbing the decision symbolically may not be the desired model. In such cases, explicit boundary classification may be preferable:

$$\text{Inside},\qquad \text{Outside},\qquad \text{Boundary} \tag{21.8.27}$$

The correct policy depends on the algorithmic purpose. Mesh traversal may need deterministic ownership. Conservative remapping may need exact boundary inclusion. Delaunay triangulation may need a unique combinatorial structure. Constructive solid geometry may need topologically consistent Boolean classification.

The central requirement is consistency. If a point is classified as lying on a boundary in one part of the code, related operations must respect that classification. If a tie is broken lexicographically in one predicate, compatible predicates must use the same convention. If a tolerance merges two vertices, subsequent edge and face reconstruction must treat them as the same entity. Robustness therefore cannot be localized to one determinant formula alone. It is a policy for the complete geometric computation.

### Rust Implementation

Following the discussion in Section 21.8.3 on degeneracy handling, symbolic perturbation, and consistent geometric classification, Program 21.8.2 provides a practical implementation of robust segment-intersection classification with explicit degeneracy policies and deterministic symbolic tie-breaking. In computational geometry, detecting degeneracy is only part of the problem; algorithms must also define how degenerate configurations are interpreted and propagated through later computations. This program demonstrates a structured approach to geometric robustness by separating raw orientation evaluation, tolerance-based classification, symbolic tie-breaking, and final intersection interpretation into distinct computational stages. The implementation distinguishes proper crossings, endpoint contact, collinear overlap, and nonintersection as separate geometric states, thereby illustrating the broader principle emphasized throughout Section 21.8: robustness is not merely a property of determinant formulas, but of the entire geometric decision pipeline. By combining explicit degeneracy handling with deterministic symbolic policies, the program provides a reliable foundation for mesh processing, clipping, Boolean geometry, and topologically consistent geometric computation.

At the core of the implementation are the `Point2`, `Segment`, `Sign`, and `SegmentIntersection` structures, which separate geometric coordinates, geometric primitives, predicate classifications, and final intersection states. The `Point2` structure stores planar coordinates together with stable integer identifiers that are later used for symbolic tie-breaking. The `Segment` structure represents an oriented line segment through its endpoints. The `Sign` enumeration provides explicit orientation classifications, while the `SegmentIntersection` enumeration distinguishes between the geometric states `None`, `ProperPoint`, `EndpointPoint`, and `CollinearOverlap`. This explicit-classification structure directly reflects the degeneracy-policy philosophy introduced in Equation (21.8.27), where boundary and degenerate states should be represented explicitly rather than hidden behind undocumented boolean logic.

The `orient2d_value` function evaluates the signed orientation determinant associated with the predicate,

$$\operatorname{orient2d}(a,b,c)$$

This determinant corresponds to the collinearity condition introduced in Equation (21.8.20). Positive determinant values indicate counterclockwise orientation, negative values indicate clockwise orientation, and zero indicates exact collinearity. The determinant forms the geometric foundation of all subsequent segment-intersection classifications.

The `sign_with_tolerance` function implements tolerance-based orientation classification. Instead of testing only for exact floating-point zero, the function introduces a tolerance interval around the origin and classifies sufficiently small determinant magnitudes as degenerate. This reflects the tolerance-policy concepts discussed earlier in Section 21.8.2. The use of explicit tolerance handling helps stabilize classifications in the presence of roundoff and cancellation effects.

The `symbolic_orientation_tie_break` function implements a deterministic symbolic perturbation policy for exactly degenerate configurations. When the orientation determinant vanishes, the function uses the integer identifiers associated with the points to produce a consistent symbolic orientation ordering. This reflects the symbolic perturbation framework introduced in Equations (21.8.24)–(21.8.26), where infinitesimal perturbations conceptually remove degeneracies while preserving deterministic behavior. The implementation avoids introducing explicit floating-point perturbations and instead simulates a symbolic ordering policy through stable combinatorial rules.

The `permutation_is_even` helper function computes the parity of the identifier ordering associated with the symbolic perturbation policy. This parity determines whether the symbolic orientation is treated as positive or negative in degenerate configurations. Although mathematically simple, this function illustrates the essential requirement emphasized in Section 21.8.3: symbolic tie-breaking must remain globally consistent across the geometric pipeline.

The `orientation_policy` function combines tolerance-based orientation testing with symbolic perturbation. The function first evaluates the determinant using the tolerance policy. If the result is degenerate and symbolic tie-breaking is enabled, the symbolic perturbation rule is applied to obtain a deterministic orientation sign. This layered structure reflects the broader robustness principle that degeneracy handling should be separated from the underlying numerical predicate evaluation.

The `point_equal` and `coordinate_order` helper functions provide geometric comparison utilities required for overlap detection and interval ordering. These functions implement tolerance-aware equality tests and lexicographic coordinate ordering, both of which are necessary for consistent classification of collinear segment configurations.

The `point_on_segment` function determines whether a query point lies on a line segment. The function first verifies collinearity using the orientation predicate and then applies interval tests to determine whether the point lies within the segment bounds. This two-stage structure reflects the broader computational principle introduced in Equation (21.8.42), where coarse geometric predicates are followed by exact local verification.

The `line_intersection_point` function computes the intersection point of two nonparallel lines using the determinant-based analytic intersection formula. The function is applied only after the orientation predicates have already established that the segments properly cross. This separation between predicate classification and geometric reconstruction is important because it prevents unstable geometric calculations from being performed on invalid configurations.

The `collinear_overlap` function implements explicit degeneracy handling for collinear segment configurations. Instead of treating collinear overlap as a failure case, the function reconstructs the overlapping interval by sorting endpoints lexicographically and testing interval inclusion conditions. This explicit overlap reconstruction demonstrates the approach advocated in Section 21.8.3, where degenerate cases are handled through dedicated geometric logic rather than hidden numerical tolerances.

The `segment_intersection` function implements the full robust segment-intersection pipeline. The function first evaluates the orientation predicates associated with both segment pairs and then classifies the configuration into one of several explicit geometric states. Proper crossings, endpoint contacts, collinear overlaps, and disjoint cases are treated separately. This structure reflects the explicit-case handling philosophy discussed in the section, where degeneracies are interpreted geometrically rather than suppressed numerically.

The helper functions `print_point`, `print_segment`, `print_intersection`, `print_case`, and `print_symbolic_demo` separate diagnostic reporting from geometric computation. These functions display geometric configurations, predicate outcomes, symbolic classifications, and final intersection states in a structured form suitable for debugging and numerical validation. The resulting output makes the interaction between geometric degeneracy and symbolic classification directly visible.

The `main` function serves to demonstrate robust degeneracy handling across a representative collection of segment configurations. It begins by constructing examples of proper crossings, endpoint contacts, collinear overlaps, disjoint collinear segments, and separated parallel segments. Each configuration is then processed using the robust segment-intersection pipeline, producing explicit geometric classifications. The program concludes with a symbolic tie-breaking demonstration using an exactly collinear point triple, thereby illustrating how deterministic symbolic perturbation removes degeneracy ambiguity while preserving consistent geometric behavior.

```rust
// Program 21.8.2: Degeneracy Handling, Segment Intersection,
// and Symbolic Tie-Breaking
//
// Problem statement:
// Demonstrate explicit degeneracy handling for two-dimensional segment
// intersection. The program distinguishes no intersection, proper crossing,
// endpoint contact, and collinear overlap. It also includes a deterministic
// symbolic tie-breaking orientation policy for exactly collinear triples,
// illustrating the degeneracy-policy ideas in Section 21.8.3.

use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Point2 {
    id: usize,
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    a: Point2,
    b: Point2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sign {
    Positive,
    Negative,
    Zero,
}

#[derive(Clone, Debug)]
enum SegmentIntersection {
    None,
    ProperPoint(Point2),
    EndpointPoint(Point2),
    CollinearOverlap(Point2, Point2),
}

fn orient2d_value(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn sign_with_tolerance(value: f64, tolerance: f64) -> Sign {
    if value > tolerance {
        Sign::Positive
    } else if value < -tolerance {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

fn symbolic_orientation_tie_break(a: Point2, b: Point2, c: Point2) -> Sign {
    let mut ids = [a.id, b.id, c.id];
    ids.sort();

    if ids[0] == ids[1] || ids[1] == ids[2] {
        return Sign::Zero;
    }

    let parity_even = permutation_is_even([a.id, b.id, c.id], ids);

    if parity_even {
        Sign::Positive
    } else {
        Sign::Negative
    }
}

fn permutation_is_even(original: [usize; 3], sorted: [usize; 3]) -> bool {
    let mut positions = [0usize; 3];

    for i in 0..3 {
        for j in 0..3 {
            if original[i] == sorted[j] {
                positions[i] = j;
                break;
            }
        }
    }

    let mut inversions = 0usize;

    for i in 0..3 {
        for j in (i + 1)..3 {
            if positions[i] > positions[j] {
                inversions += 1;
            }
        }
    }

    inversions % 2 == 0
}

fn orientation_policy(
    a: Point2,
    b: Point2,
    c: Point2,
    tolerance: f64,
    use_symbolic_tie_break: bool,
) -> Sign {
    let det = orient2d_value(a, b, c);
    let sign = sign_with_tolerance(det, tolerance);

    if sign == Sign::Zero && use_symbolic_tie_break {
        symbolic_orientation_tie_break(a, b, c)
    } else {
        sign
    }
}

fn point_equal(a: Point2, b: Point2, tolerance: f64) -> bool {
    (a.x - b.x).abs() <= tolerance && (a.y - b.y).abs() <= tolerance
}

fn coordinate_order(a: Point2, b: Point2) -> Ordering {
    match a.x.partial_cmp(&b.x).unwrap() {
        Ordering::Equal => a.y.partial_cmp(&b.y).unwrap(),
        other => other,
    }
}

fn point_on_segment(a: Point2, b: Point2, p: Point2, tolerance: f64) -> bool {
    if sign_with_tolerance(orient2d_value(a, b, p), tolerance) != Sign::Zero {
        return false;
    }

    p.x >= a.x.min(b.x) - tolerance
        && p.x <= a.x.max(b.x) + tolerance
        && p.y >= a.y.min(b.y) - tolerance
        && p.y <= a.y.max(b.y) + tolerance
}

fn line_intersection_point(s1: Segment, s2: Segment) -> Option<Point2> {
    let x1 = s1.a.x;
    let y1 = s1.a.y;
    let x2 = s1.b.x;
    let y2 = s1.b.y;

    let x3 = s2.a.x;
    let y3 = s2.a.y;
    let x4 = s2.b.x;
    let y4 = s2.b.y;

    let denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

    if denominator.abs() <= 1.0e-14 {
        return None;
    }

    let det1 = x1 * y2 - y1 * x2;
    let det2 = x3 * y4 - y3 * x4;

    let px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denominator;
    let py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denominator;

    Some(Point2 {
        id: usize::MAX,
        x: px,
        y: py,
    })
}

fn collinear_overlap(
    s1: Segment,
    s2: Segment,
    tolerance: f64,
) -> SegmentIntersection {
    let mut points = [s1.a, s1.b, s2.a, s2.b];

    points.sort_by(|p, q| coordinate_order(*p, *q));

    let left = points[1];
    let right = points[2];

    if point_on_segment(s1.a, s1.b, left, tolerance)
        && point_on_segment(s1.a, s1.b, right, tolerance)
        && point_on_segment(s2.a, s2.b, left, tolerance)
        && point_on_segment(s2.a, s2.b, right, tolerance)
    {
        if point_equal(left, right, tolerance) {
            SegmentIntersection::EndpointPoint(left)
        } else {
            SegmentIntersection::CollinearOverlap(left, right)
        }
    } else {
        SegmentIntersection::None
    }
}

fn segment_intersection(
    s1: Segment,
    s2: Segment,
    tolerance: f64,
) -> SegmentIntersection {
    let o1 = sign_with_tolerance(orient2d_value(s1.a, s1.b, s2.a), tolerance);
    let o2 = sign_with_tolerance(orient2d_value(s1.a, s1.b, s2.b), tolerance);
    let o3 = sign_with_tolerance(orient2d_value(s2.a, s2.b, s1.a), tolerance);
    let o4 = sign_with_tolerance(orient2d_value(s2.a, s2.b, s1.b), tolerance);

    if o1 == Sign::Zero
        && o2 == Sign::Zero
        && o3 == Sign::Zero
        && o4 == Sign::Zero
    {
        return collinear_overlap(s1, s2, tolerance);
    }

    if o1 == Sign::Zero && point_on_segment(s1.a, s1.b, s2.a, tolerance) {
        return SegmentIntersection::EndpointPoint(s2.a);
    }

    if o2 == Sign::Zero && point_on_segment(s1.a, s1.b, s2.b, tolerance) {
        return SegmentIntersection::EndpointPoint(s2.b);
    }

    if o3 == Sign::Zero && point_on_segment(s2.a, s2.b, s1.a, tolerance) {
        return SegmentIntersection::EndpointPoint(s1.a);
    }

    if o4 == Sign::Zero && point_on_segment(s2.a, s2.b, s1.b, tolerance) {
        return SegmentIntersection::EndpointPoint(s1.b);
    }

    let s1_straddles = o1 != o2;
    let s2_straddles = o3 != o4;

    if s1_straddles && s2_straddles {
        if let Some(p) = line_intersection_point(s1, s2) {
            return SegmentIntersection::ProperPoint(p);
        }
    }

    SegmentIntersection::None
}

fn print_point(label: &str, p: Point2) {
    if p.id == usize::MAX {
        println!("{label}({:>10.6}, {:>10.6})", p.x, p.y);
    } else {
        println!("{label}p_{} = ({:>10.6}, {:>10.6})", p.id, p.x, p.y);
    }
}

fn print_segment(label: &str, s: Segment) {
    println!("{label}");
    print_point("  a = ", s.a);
    print_point("  b = ", s.b);
}

fn print_intersection(result: &SegmentIntersection) {
    match result {
        SegmentIntersection::None => {
            println!("intersection type = None");
        }
        SegmentIntersection::ProperPoint(p) => {
            println!("intersection type = ProperPoint");
            print_point("intersection = ", *p);
        }
        SegmentIntersection::EndpointPoint(p) => {
            println!("intersection type = EndpointPoint");
            print_point("contact point = ", *p);
        }
        SegmentIntersection::CollinearOverlap(a, b) => {
            println!("intersection type = CollinearOverlap");
            print_point("overlap start = ", *a);
            print_point("overlap end   = ", *b);
        }
    }
}

fn print_case(label: &str, s1: Segment, s2: Segment, tolerance: f64) {
    println!("{label}");
    println!("{}", "-".repeat(label.len()));

    print_segment("segment 1", s1);
    print_segment("segment 2", s2);

    let result = segment_intersection(s1, s2, tolerance);
    print_intersection(&result);

    println!();
}

fn print_symbolic_demo(a: Point2, b: Point2, c: Point2, tolerance: f64) {
    println!("Symbolic Tie-Breaking Demonstration");
    println!("-----------------------------------");
    print_point("a = ", a);
    print_point("b = ", b);
    print_point("c = ", c);

    let raw = sign_with_tolerance(orient2d_value(a, b, c), tolerance);
    let symbolic = orientation_policy(a, b, c, tolerance, true);

    println!("tolerance orientation = {:?}", raw);
    println!("symbolic orientation  = {:?}", symbolic);
    println!(
        "The symbolic result is deterministic and depends only on the point identifiers."
    );
    println!();
}

fn main() {
    let tolerance = 1.0e-12;

    println!("Degeneracy Handling and Segment Intersection");
    println!("============================================\n");

    println!("Predicate Policy");
    println!("----------------");
    println!("tolerance = {:>12.6e}", tolerance);
    println!("degenerate cases are classified explicitly");
    println!();

    let proper_crossing = (
        Segment {
            a: Point2 {
                id: 0,
                x: 0.0,
                y: 0.0,
            },
            b: Point2 {
                id: 1,
                x: 2.0,
                y: 2.0,
            },
        },
        Segment {
            a: Point2 {
                id: 2,
                x: 0.0,
                y: 2.0,
            },
            b: Point2 {
                id: 3,
                x: 2.0,
                y: 0.0,
            },
        },
    );

    let endpoint_contact = (
        Segment {
            a: Point2 {
                id: 4,
                x: 0.0,
                y: 0.0,
            },
            b: Point2 {
                id: 5,
                x: 1.0,
                y: 1.0,
            },
        },
        Segment {
            a: Point2 {
                id: 5,
                x: 1.0,
                y: 1.0,
            },
            b: Point2 {
                id: 6,
                x: 2.0,
                y: 0.0,
            },
        },
    );

    let collinear_overlap_case = (
        Segment {
            a: Point2 {
                id: 7,
                x: 0.0,
                y: 0.0,
            },
            b: Point2 {
                id: 8,
                x: 3.0,
                y: 0.0,
            },
        },
        Segment {
            a: Point2 {
                id: 9,
                x: 1.0,
                y: 0.0,
            },
            b: Point2 {
                id: 10,
                x: 2.5,
                y: 0.0,
            },
        },
    );

    let disjoint_collinear = (
        Segment {
            a: Point2 {
                id: 11,
                x: 0.0,
                y: 1.0,
            },
            b: Point2 {
                id: 12,
                x: 1.0,
                y: 1.0,
            },
        },
        Segment {
            a: Point2 {
                id: 13,
                x: 2.0,
                y: 1.0,
            },
            b: Point2 {
                id: 14,
                x: 3.0,
                y: 1.0,
            },
        },
    );

    let no_intersection = (
        Segment {
            a: Point2 {
                id: 15,
                x: 0.0,
                y: 0.0,
            },
            b: Point2 {
                id: 16,
                x: 1.0,
                y: 0.0,
            },
        },
        Segment {
            a: Point2 {
                id: 17,
                x: 0.0,
                y: 1.0,
            },
            b: Point2 {
                id: 18,
                x: 1.0,
                y: 1.0,
            },
        },
    );

    print_case(
        "Proper Crossing",
        proper_crossing.0,
        proper_crossing.1,
        tolerance,
    );

    print_case(
        "Endpoint Contact",
        endpoint_contact.0,
        endpoint_contact.1,
        tolerance,
    );

    print_case(
        "Collinear Overlap",
        collinear_overlap_case.0,
        collinear_overlap_case.1,
        tolerance,
    );

    print_case(
        "Disjoint Collinear Segments",
        disjoint_collinear.0,
        disjoint_collinear.1,
        tolerance,
    );

    print_case(
        "Separated Parallel Segments",
        no_intersection.0,
        no_intersection.1,
        tolerance,
    );

    print_symbolic_demo(
        Point2 {
            id: 21,
            x: 0.0,
            y: 0.0,
        },
        Point2 {
            id: 22,
            x: 1.0,
            y: 1.0,
        },
        Point2 {
            id: 23,
            x: 2.0,
            y: 2.0,
        },
        tolerance,
    );
}
```

Program 21.8.2 demonstrates how robust geometric algorithms must combine numerical predicates with explicit degeneracy policies and consistent geometric interpretation. The implementation separates orientation evaluation, symbolic tie-breaking, local containment verification, and final intersection classification into distinct computational stages, thereby producing a robust and transparent geometric-processing pipeline.

The examples illustrate several important robustness behaviors. Proper segment crossings are reconstructed geometrically after orientation predicates certify the crossing configuration. Endpoint contacts are treated as distinct geometric states rather than accidental special cases. Collinear overlap configurations are reconstructed explicitly through interval reasoning instead of being discarded as numerical degeneracies. The symbolic tie-breaking example further demonstrates how deterministic perturbation policies can eliminate ambiguity in exactly degenerate configurations while preserving consistent combinatorial behavior.

The implementation also highlights the broader computational philosophy emphasized throughout Section 21.8. Robustness cannot be confined to a single determinant evaluation. Instead, it must propagate consistently through the entire geometric pipeline, including intersection classification, overlap reconstruction, topological interpretation, and downstream geometric processing. Explicit enums, deterministic policies, and structured geometric states provide a safer foundation for large geometric systems than undocumented floating-point heuristics.

More broadly, the program demonstrates that degeneracy handling is fundamentally an algorithmic policy question rather than merely a numerical issue. Different applications require different interpretations of degenerate geometry. Symbolic perturbation may be appropriate for combinatorial triangulation algorithms, while explicit boundary classification may be preferable for physical simulation and conservative remapping. The essential requirement is that the chosen policy remain globally consistent throughout the geometric computation. This principle forms the foundation of reliable modern computational geometry kernels and robust scientific geometric software.

## 21.8.4. From Predicate Robustness to Pipeline Robustness

Predicate robustness ensures that individual signs are correct. Pipeline robustness ensures that the entire geometric data structure remains valid after many local operations. This distinction is essential. A mesh arrangement algorithm may use exact orientation and intersection predicates, but still fail if it does not merge duplicate vertices consistently, sort intersection points along edges correctly, or reconstruct faces with the proper adjacency. Similarly, a constructive solid geometry algorithm may classify individual points correctly but still produce invalid topology if boundary fragments are not assembled coherently.

It is useful to distinguish three levels of robustness:

$$\text{mathematical robustness},\qquad\text{predicate robustness},\\ \qquad\text{pipeline robustness} \tag{21.8.28}$$

Mathematical robustness means that the algorithm is well defined on degenerate input. Predicate robustness means that the branch decisions are evaluated correctly. Pipeline robustness means that all local decisions produce a globally consistent topological object.

Mesh intersection illustrates this hierarchy. Let two meshes be:

$$\mathcal{A}=\{A_i\}_{i=1}^{N_A},\qquad\mathcal{B}=\{B_j\}_{j=1}^{N_B} \tag{21.8.29}$$

The arrangement is built from nonempty intersections:

$$C_{ij}=A_i\cap B_j \tag{21.8.30}$$

Even if every local intersection $A_i\cap B_j$ is computed accurately, the arrangement can fail unless coincident intersection vertices are identified, edges are split consistently, and the induced faces and cells form a coherent complex. The required output is not merely a list of intersection points, but a topological structure with valid incidence relations:

$$\text{vertices}\longrightarrow\text{edges}\longrightarrow\text{faces}\longrightarrow\text{cells}\tag{21.8.31}$$

Exact mesh-arrangement and exact mesh-CSG methods emphasize precisely this point: robust geometry at scale requires both exact local decisions and consistent combinatorial reconstruction (Guo and Fu, 2024; Lévy, 2025).

The same principle applies to conservative remapping. Suppose a quantity $u$ is transferred from a source mesh to a target mesh. For a source cell $A_i$ and target cells $B_j$, conservation may require contributions over overlap regions:

$$\int_{A_i} u(x)\,dx = \sum_j\int_{A_i\cap B_j} u(x)\,dx \tag{21.8.32}$$

up to quadrature and discretization error. If the computed overlaps $A_i\cap B_j$ contain gaps or overlaps because of geometric inconsistency, conservation can fail. The numerical method may appear to have a flux or quadrature problem, but the underlying cause is geometric.

Cut-cell and volume-of-fluid methods provide another example. A cell $K$ may be cut by an interface $\Gamma$, producing a material fraction:

$$\alpha_K=\frac{|K\cap D|}{|K|} \tag{21.8.33}$$

If the interface-cell intersection is not reconstructed consistently, $\alpha_K$ may be inaccurate or even nonphysical:

$$\alpha_K<0\qquad\text{or}\qquad\alpha_K>1 \tag{21.8.34}$$

Such failures are not acceptable in conservative physical simulations. Robust polytope intersection algorithms for arbitrary-grid volume-of-fluid methods therefore address both geometric accuracy and topological consistency (López and Hernández, 2024).

Thus, robustness should be viewed as a pipeline property. The predicate layer must certify signs. The construction layer must represent new geometric objects consistently. The topology layer must maintain incidence and adjacency. The numerical layer must receive geometrically valid cells, measures, and classifications. A failure at any layer can invalidate the final scientific computation.

## 21.8.5. Implementation Principles for Robust Geometry in Rust

A robust implementation should make geometric policy explicit. The code should not scatter unrelated tolerances across orientation tests, point containment, clipping, and mesh traversal. Instead, it should define a predicate kernel and require higher-level algorithms to use it consistently:

$$\text{numeric kernel}\longrightarrow\text{certified predicates}\longrightarrow\text{classification types}\\ \longrightarrow\text{topological algorithms} \tag{21.8.35}$$

The numeric kernel evaluates determinants, distances, dot products, and affine coordinates. The certified predicate layer returns signs or classifications. The classification layer distinguishes cases such as inside, outside, boundary, proper intersection, endpoint contact, overlap, valid orientation, inverted orientation, and degeneracy. The topological layer uses these classifications to update meshes, arrangements, triangulations, or search structures.

In Rust, predicate results should be represented by explicit enums rather than raw floating-point values or booleans. For example, an orientation predicate should conceptually return:

$$\text{Positive},\qquad \text{Negative},\qquad \text{Zero} \tag{21.8.36}$$

and a containment predicate should return:

$$\text{Inside},\qquad \text{Outside},\qquad \text{Boundary} \tag{21.8.37}$$

A segment-intersection predicate should distinguish:

$$\text{None},\qquad\text{ProperPoint},\qquad\text{EndpointPoint},\qquad\text{CollinearOverlap} \tag{21.8.38}$$

Such distinctions prevent higher-level code from losing information that is essential for robust topology.

For algorithms that use spatial indexes, bounding boxes, or grids, robustness requires a two-stage pattern:

$$\text{coarse candidate generation}\quad+\quad\text{exact local verification} \tag{21.8.39}$$

A bounding volume hierarchy may return candidate triangles. A grid may return candidate particles. A $k$-d tree may return candidate cells. These candidates should then be checked by exact or certified local predicates before final classification. Spatial search should accelerate geometry, not replace it.

Testing must include degenerate and near-degenerate cases. A robust geometry test suite should contain collinear triples, nearly collinear triples, coplanar tetrahedra, nearly coplanar tetrahedra, cocircular point sets, cospherical point sets, overlapping segments, endpoint contacts, boundary point-location queries, zero-volume cells, inverted cells, and mesh intersections with shared vertices or edges. The goal is not only to check arithmetic outputs, but to verify that the final classifications and topological structures are consistent.

A practical Rust design for this chapter can therefore follow this pattern:

$$\text{immutable primitives}\longrightarrow\text{index-based topology}\longrightarrow\text{certified predicates} \\ \longrightarrow\text{validated local updates}\longrightarrow\text{global consistency checks}\tag{21.8.40}$$

Immutable primitives make low-level geometry predictable. Index-based topology avoids unsafe pointer-rich mutation. Certified predicates protect branch decisions. Validated local updates ensure that edge flips, cavity retriangulations, clipping operations, and mesh intersections preserve local invariants. Global checks verify that the resulting mesh or arrangement is coherent.

The main conclusion is that robust geometry is numerical correctness. A PDE solver assembled on an invalid mesh, a remapping scheme using inconsistent cell overlaps, a point-location method with nondeterministic boundary ownership, or a Delaunay triangulation corrupted by an incircle error may fail before the intended numerical method is even tested. The geometric layer supplies the discrete structure on which the numerical method relies. For this reason, finite-precision robustness is not an optional refinement of computational geometry. It is the condition under which geometric algorithms become reliable components of scientific computing.

### Rust Implementation

Following the discussion in Sections 21.8.4 and 21.8.5 on pipeline robustness, topological consistency, and robust geometric software design, Program 21.8.3 provides a practical implementation of a robust polygon-clipping pipeline using explicit predicate classifications, duplicate-vertex merging, and global consistency validation. In robust computational geometry, correct determinant signs alone are insufficient to guarantee valid geometric output. A geometric pipeline must also ensure that newly constructed vertices, edges, and polygonal regions remain topologically coherent after repeated local operations. This program demonstrates the layered robustness architecture introduced in Equation (21.8.35), where numeric kernels, certified predicates, classification types, and topological updates are organized into separate computational stages. Using half-plane clipping as a representative geometric operation, the implementation combines exact local classification with postprocessing validation and consistency checks to produce geometrically and topologically valid output polygons. The framework illustrates the broader principle emphasized throughout Section 21.8: robust geometry is not merely about individual predicates, but about preserving global consistency across the complete geometric-processing pipeline.

At the core of the implementation are the `Point2`, `HalfPlane`, `Polygon`, `Sign`, and `SideClassification` structures, which separate geometric primitives, predicate classifications, and topological representations. The `Point2` structure stores planar coordinates, while the `HalfPlane` structure represents a clipping region through a normal vector and scalar offset. The `Polygon` structure stores polygon vertices in cyclic order, thereby representing the topological boundary of a two-dimensional region. The `Sign` and `SideClassification` enumerations implement explicit classification states for predicate evaluation and geometric containment. This explicit classification structure directly reflects the robustness philosophy introduced in Equations (21.8.36) and (21.8.37), where predicates should return structured geometric states rather than undocumented booleans.

The `PipelineReport` structure provides a global diagnostic summary of the clipping pipeline. Instead of merely returning the clipped polygon, the implementation records information about input and output vertex counts, duplicate-vertex removal, polygon areas, boundary classifications, and final topological validity. This diagnostic structure reflects the broader pipeline-robustness concepts introduced in Equation (21.8.28), where robustness must be evaluated not only at the predicate level, but across the complete geometric-processing pipeline.

The `signed_distance_to_half_plane` function evaluates the signed distance-like quantity associated with the half-plane inequality $n\cdot x-\beta \le 0$. This quantity forms the basis of the point-classification logic used throughout the clipping pipeline. Negative values correspond to points inside the half-plane, positive values correspond to points outside, and near-zero values correspond to boundary configurations.

The `sign_with_tolerance` function implements tolerance-aware numerical classification of floating-point quantities. Instead of relying on exact floating-point zero comparisons, the function introduces a tolerance interval around the origin and classifies sufficiently small values as degenerate. This approach reflects the tolerance-policy concepts discussed earlier in Section 21.8.2 and supports stable geometric classification near boundaries.

The `classify_point_half_plane` function implements explicit geometric classification relative to the clipping half-plane. Using the tolerance-aware sign policy, the function returns one of the states `Inside`, `Outside`, or `Boundary`. This explicit-state representation directly reflects the classification framework introduced in Equation (21.8.37). Boundary points are treated as a distinct geometric category rather than implicitly merged with either interior or exterior states.

The `segment_half_plane_intersection` function computes the geometric intersection point between a segment and the clipping boundary. The function first evaluates the signed distance values at the segment endpoints and then computes the parametric intersection point using affine interpolation. This geometric reconstruction stage is performed only after the classification logic has determined that an intersection must exist, thereby separating geometric predicates from geometric construction.

The `clip_polygon_raw` function implements the core half-plane clipping algorithm. The function traverses the polygon boundary edge by edge and applies the classification policy to determine how each edge contributes to the clipped polygon. Interior-to-interior edges contribute their endpoint directly, while edges crossing the clipping boundary contribute newly computed intersection points. Exterior-to-exterior edges contribute nothing. This edge-processing structure reflects the geometric reconstruction philosophy introduced in Equation (21.8.31), where valid topological output must be assembled from locally classified geometric fragments.

The `squared_distance` helper function computes the squared Euclidean distance between two points. This function supports duplicate-vertex detection without requiring unnecessary square-root operations, thereby improving numerical efficiency and consistency.

The `merge_near_duplicate_vertices` function implements one of the key pipeline-robustness operations emphasized in Section 21.8.4: consistent identification and merging of coincident geometric entities. Clipping and intersection operations often generate duplicate or nearly duplicate vertices because adjacent edges may independently produce geometrically equivalent intersection points. The function removes such near-duplicates using a tolerance-aware geometric comparison policy. This stage is essential because even geometrically correct local operations can produce globally inconsistent topology if duplicate vertices are not merged consistently.

The `polygon_signed_area` function evaluates the oriented area of a polygon using the shoelace formula. The sign of the area determines polygon orientation, while the magnitude determines geometric measure. This area computation provides a global consistency diagnostic because degenerate or self-inconsistent polygons may produce zero or invalid areas.

The `polygon_is_valid` function performs a basic topological consistency check on the resulting polygon. The function verifies that the polygon contains at least three vertices and that its area exceeds the numerical tolerance threshold. This validation stage reflects the broader robustness philosophy introduced in Equation (21.8.40), where local geometric updates must be followed by global consistency verification.

The `classify_polygon_vertices` function performs a postprocessing geometric audit of the clipped polygon by classifying all output vertices relative to the clipping half-plane. This verification step ensures that the resulting polygon satisfies the intended geometric classification policy and demonstrates how robust pipelines should validate the consistency of their outputs.

The `robust_clip_pipeline` function combines local clipping operations, duplicate-vertex merging, geometric validation, and diagnostic reporting into a complete robust geometric-processing pipeline. The function first performs raw clipping, then merges coincident vertices, computes geometric measures, validates the output polygon, and assembles the global consistency report. This layered structure directly reflects the pipeline-robustness architecture introduced in Equation (21.8.35).

The helper functions `print_point`, `print_polygon`, `print_half_plane`, and `print_report` separate diagnostic reporting from geometric computation. These functions present polygon geometry, clipping configuration, and pipeline-consistency information in a structured form suitable for debugging and numerical validation. The resulting output makes the interaction between local geometric operations and global topological consistency directly visible.

The `main` function serves to demonstrate robust polygon clipping using a representative convex polygon and a clipping half-plane. The program begins by constructing a rectangular polygon together with a half-plane boundary corresponding to the inequality (x\\le2.5). The clipping pipeline is then executed in multiple stages: raw clipping, duplicate-vertex merging, polygon validation, and consistency reporting. The resulting output demonstrates how robust geometric pipelines combine predicate classification, geometric reconstruction, topological cleanup, and global verification to produce valid geometric structures suitable for downstream scientific computation.

```rust
// Program 21.8.3: Robust Polygon Clipping Pipeline with Predicate Kernel,
// Vertex Merging, and Consistency Checks
//
// Problem statement:
// Demonstrate pipeline robustness by clipping a convex polygon against a
// half-plane using an explicit predicate kernel, structured classifications,
// duplicate-vertex merging, and global consistency checks. The program follows
// the robustness pipeline ideas in Section 21.8.4 and the Rust implementation
// principles in Section 21.8.5.

#[derive(Clone, Copy, Debug)]
struct Point2 {
    x: f64,
    y: f64,
}

#[derive(Clone, Copy, Debug)]
struct HalfPlane {
    nx: f64,
    ny: f64,
    beta: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sign {
    Positive,
    Negative,
    Zero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SideClassification {
    Inside,
    Outside,
    Boundary,
}

#[derive(Clone, Debug)]
struct Polygon {
    vertices: Vec<Point2>,
}

#[derive(Clone, Debug)]
struct PipelineReport {
    input_vertices: usize,
    raw_output_vertices: usize,
    merged_output_vertices: usize,
    input_area: f64,
    output_area: f64,
    duplicate_vertices_removed: usize,
    boundary_vertices: usize,
    inside_vertices: usize,
    outside_vertices: usize,
    valid_polygon: bool,
}

fn signed_distance_to_half_plane(p: Point2, h: HalfPlane) -> f64 {
    h.nx * p.x + h.ny * p.y - h.beta
}

fn sign_with_tolerance(value: f64, tolerance: f64) -> Sign {
    if value > tolerance {
        Sign::Positive
    } else if value < -tolerance {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

fn classify_point_half_plane(
    p: Point2,
    h: HalfPlane,
    tolerance: f64,
) -> SideClassification {
    match sign_with_tolerance(signed_distance_to_half_plane(p, h), tolerance) {
        Sign::Negative | Sign::Zero => {
            if signed_distance_to_half_plane(p, h).abs() <= tolerance {
                SideClassification::Boundary
            } else {
                SideClassification::Inside
            }
        }
        Sign::Positive => SideClassification::Outside,
    }
}

fn segment_half_plane_intersection(
    a: Point2,
    b: Point2,
    h: HalfPlane,
) -> Option<Point2> {
    let fa = signed_distance_to_half_plane(a, h);
    let fb = signed_distance_to_half_plane(b, h);
    let denominator = fa - fb;

    if denominator.abs() <= 1.0e-14 {
        return None;
    }

    let t = fa / (fa - fb);

    if !(0.0..=1.0).contains(&t) {
        return None;
    }

    Some(Point2 {
        x: a.x + t * (b.x - a.x),
        y: a.y + t * (b.y - a.y),
    })
}

fn clip_polygon_raw(
    polygon: &Polygon,
    h: HalfPlane,
    tolerance: f64,
) -> Polygon {
    if polygon.vertices.is_empty() {
        return Polygon {
            vertices: Vec::new(),
        };
    }

    let mut output = Vec::new();
    let n = polygon.vertices.len();

    for i in 0..n {
        let current = polygon.vertices[i];
        let next = polygon.vertices[(i + 1) % n];

        let current_class = classify_point_half_plane(current, h, tolerance);
        let next_class = classify_point_half_plane(next, h, tolerance);

        let current_inside = matches!(
            current_class,
            SideClassification::Inside | SideClassification::Boundary
        );

        let next_inside = matches!(
            next_class,
            SideClassification::Inside | SideClassification::Boundary
        );

        match (current_inside, next_inside) {
            (true, true) => {
                output.push(next);
            }
            (true, false) => {
                if let Some(p) = segment_half_plane_intersection(current, next, h) {
                    output.push(p);
                }
            }
            (false, true) => {
                if let Some(p) = segment_half_plane_intersection(current, next, h) {
                    output.push(p);
                }
                output.push(next);
            }
            (false, false) => {}
        }
    }

    Polygon { vertices: output }
}

fn squared_distance(a: Point2, b: Point2) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

fn merge_near_duplicate_vertices(
    polygon: &Polygon,
    tolerance: f64,
) -> Polygon {
    if polygon.vertices.is_empty() {
        return Polygon {
            vertices: Vec::new(),
        };
    }

    let tol2 = tolerance * tolerance;
    let mut merged = Vec::new();

    for &p in &polygon.vertices {
        if merged
            .last()
            .map(|&q| squared_distance(p, q) > tol2)
            .unwrap_or(true)
        {
            merged.push(p);
        }
    }

    if merged.len() > 1 {
        let first = merged[0];
        let last = *merged.last().unwrap();

        if squared_distance(first, last) <= tol2 {
            merged.pop();
        }
    }

    Polygon { vertices: merged }
}

fn polygon_signed_area(polygon: &Polygon) -> f64 {
    let n = polygon.vertices.len();

    if n < 3 {
        return 0.0;
    }

    let mut area2 = 0.0;

    for i in 0..n {
        let p = polygon.vertices[i];
        let q = polygon.vertices[(i + 1) % n];

        area2 += p.x * q.y - p.y * q.x;
    }

    0.5 * area2
}

fn polygon_is_valid(polygon: &Polygon, tolerance: f64) -> bool {
    polygon.vertices.len() >= 3 && polygon_signed_area(polygon).abs() > tolerance
}

fn classify_polygon_vertices(
    polygon: &Polygon,
    h: HalfPlane,
    tolerance: f64,
) -> (usize, usize, usize) {
    let mut inside = 0usize;
    let mut boundary = 0usize;
    let mut outside = 0usize;

    for &p in &polygon.vertices {
        match classify_point_half_plane(p, h, tolerance) {
            SideClassification::Inside => inside += 1,
            SideClassification::Boundary => boundary += 1,
            SideClassification::Outside => outside += 1,
        }
    }

    (inside, boundary, outside)
}

fn robust_clip_pipeline(
    input: &Polygon,
    h: HalfPlane,
    tolerance: f64,
) -> (Polygon, PipelineReport) {
    let input_area = polygon_signed_area(input).abs();

    let raw = clip_polygon_raw(input, h, tolerance);
    let raw_output_vertices = raw.vertices.len();

    let merged = merge_near_duplicate_vertices(&raw, tolerance);
    let merged_output_vertices = merged.vertices.len();

    let output_area = polygon_signed_area(&merged).abs();

    let duplicate_vertices_removed =
        raw_output_vertices.saturating_sub(merged_output_vertices);

    let (inside_vertices, boundary_vertices, outside_vertices) =
        classify_polygon_vertices(&merged, h, tolerance);

    let valid_polygon = polygon_is_valid(&merged, tolerance);

    let report = PipelineReport {
        input_vertices: input.vertices.len(),
        raw_output_vertices,
        merged_output_vertices,
        input_area,
        output_area,
        duplicate_vertices_removed,
        boundary_vertices,
        inside_vertices,
        outside_vertices,
        valid_polygon,
    };

    (merged, report)
}

fn print_point(index: usize, p: Point2) {
    println!("p_{:<2} = ({:>10.6}, {:>10.6})", index, p.x, p.y);
}

fn print_polygon(title: &str, polygon: &Polygon) {
    println!("{title}");
    println!("{}", "-".repeat(title.len()));

    if polygon.vertices.is_empty() {
        println!("empty polygon");
        println!();
        return;
    }

    for (i, p) in polygon.vertices.iter().enumerate() {
        print_point(i, *p);
    }

    println!("signed area = {:>12.8}", polygon_signed_area(polygon));
    println!("area        = {:>12.8}", polygon_signed_area(polygon).abs());
    println!();
}

fn print_half_plane(h: HalfPlane) {
    println!("Half-Plane");
    println!("----------");
    println!("n          = ({:>10.6}, {:>10.6})", h.nx, h.ny);
    println!("beta       = {:>10.6}", h.beta);
    println!("inside set = n . x - beta <= 0");
    println!();
}

fn print_report(report: &PipelineReport) {
    println!("Pipeline Consistency Report");
    println!("---------------------------");
    println!("input vertices             = {}", report.input_vertices);
    println!("raw output vertices        = {}", report.raw_output_vertices);
    println!("merged output vertices     = {}", report.merged_output_vertices);
    println!(
        "duplicate vertices removed = {}",
        report.duplicate_vertices_removed
    );
    println!("input area                 = {:>12.8}", report.input_area);
    println!("output area                = {:>12.8}", report.output_area);
    println!("inside output vertices     = {}", report.inside_vertices);
    println!("boundary output vertices   = {}", report.boundary_vertices);
    println!("outside output vertices    = {}", report.outside_vertices);
    println!("valid output polygon       = {}", report.valid_polygon);
    println!();
}

fn main() {
    let tolerance = 1.0e-12;

    let polygon = Polygon {
        vertices: vec![
            Point2 { x: 0.0, y: 0.0 },
            Point2 { x: 4.0, y: 0.0 },
            Point2 { x: 4.0, y: 3.0 },
            Point2 { x: 0.0, y: 3.0 },
        ],
    };

    let half_plane = HalfPlane {
        nx: 1.0,
        ny: 0.0,
        beta: 2.5,
    };

    println!("Robust Polygon Clipping Pipeline");
    println!("================================\n");

    println!("Predicate and Pipeline Policy");
    println!("-----------------------------");
    println!("tolerance = {:>12.6e}", tolerance);
    println!("classification uses explicit Inside, Outside, and Boundary states");
    println!("pipeline applies candidate construction, duplicate merging, and consistency checks");
    println!();

    print_polygon("Input Polygon", &polygon);
    print_half_plane(half_plane);

    let raw = clip_polygon_raw(&polygon, half_plane, tolerance);
    print_polygon("Raw Clipped Polygon Before Vertex Merging", &raw);

    let (output, report) = robust_clip_pipeline(&polygon, half_plane, tolerance);

    print_polygon("Final Clipped Polygon After Vertex Merging", &output);
    print_report(&report);

    println!("Final Interpretation");
    println!("--------------------");
    if report.outside_vertices == 0 && report.valid_polygon {
        println!("All output vertices satisfy the half-plane classification policy.");
        println!("The clipped polygon is topologically valid for this convex example.");
    } else {
        println!("The pipeline detected an invalid or inconsistent clipped polygon.");
    }
}
```

Program 21.8.3 demonstrates how robust computational geometry requires more than numerically correct predicates. The implementation combines explicit geometric classification, geometric reconstruction, duplicate-vertex merging, and global consistency verification into a complete geometric-processing pipeline capable of producing coherent polygonal output.

The example illustrates several important robustness principles. Predicate-level classification determines whether points lie inside, outside, or on the clipping boundary. Geometric reconstruction computes new intersection vertices only after valid crossing configurations are identified. Duplicate-vertex merging ensures that geometrically equivalent entities are represented consistently, while polygon validation verifies that the resulting topological structure remains meaningful after reconstruction. These stages collectively illustrate the hierarchy introduced in Equation (21.8.28): mathematical robustness, predicate robustness, and pipeline robustness.

The implementation also highlights the broader software-engineering philosophy emphasized throughout Section 21.8. Robust geometry should be organized into layered computational structures where numeric kernels, predicate classifications, geometric construction, and topological verification remain conceptually distinct. Explicit enums, immutable geometric primitives, tolerance-aware validation, and structured consistency checks produce software that is safer, easier to debug, and more resistant to topological failure.

More broadly, the program demonstrates why robust geometry is fundamentally a numerical-scientific requirement rather than a purely geometric refinement. Conservative remapping, cut-cell methods, mesh intersection, polygon clipping, and geometric reconstruction all depend on globally coherent topology and consistent geometric classification. Even when local predicates are mathematically correct, inconsistent reconstruction or topology can invalidate the final numerical method. Robust geometric pipelines therefore form part of the numerical correctness of scientific computing itself.

# 21.9. Conclusion

Throughout this chapter, we have explored the mathematical and computational foundations of geometric algorithms used in scientific computing. From basic geometric predicates to advanced structures such as Delaunay triangulations, Voronoi diagrams, and spatial-search data structures, we have seen how geometry provides the foundation for mesh generation, simulation, visualization, optimization, and computational modeling. Particular emphasis was placed on robustness, since even mathematically correct algorithms can fail when implemented using finite-precision arithmetic. By combining geometric theory with practical Rust implementations, this chapter has provided the tools necessary for constructing reliable and efficient geometric software suitable for scientific and engineering applications.

## 21.9.1. Key Takeaways

- Computational geometry is fundamentally concerned with representation and decision. Geometric algorithms operate by combining data structures with predicates that classify spatial relationships and guide computational decisions.
- Orientation, incircle, and insphere predicates form the foundation of many geometric algorithms. Their accuracy directly affects the correctness of intersection tests, triangulations, mesh operations, and spatial searches.
- Segment-intersection, clipping, and point-in-polygon algorithms provide essential tools for geometric processing pipelines used in computer graphics, simulation, geographic information systems, and scientific visualization.
- Convex hulls capture the outer structure of point sets and serve as important building blocks for many higher-level geometric algorithms and optimization procedures.
- Delaunay triangulations and Voronoi diagrams provide mathematically elegant and computationally efficient frameworks for mesh generation, interpolation, nearest-neighbor analysis, and spatial modeling.
- Spatial-search structures such as uniform grids, k-d trees, octrees, ball trees, and bounding-volume hierarchies allow geometric queries to be performed efficiently even for large datasets.
- Mesh quality directly influences the accuracy and stability of numerical simulations. Poorly shaped elements can significantly degrade the performance of computational algorithms.
- Barycentric coordinates provide a natural framework for interpolation, point-location, and geometric reasoning within simplices and mesh elements.
- Finite-precision arithmetic introduces degeneracies and classification ambiguities that must be addressed through robust predicates, filtering techniques, and consistent algorithm design.
- Rust provides an excellent platform for geometric computing by combining performance, memory safety, expressive abstractions, and reliable implementation practices.

## 21.9.2. Advice for Beginners

- When studying computational geometry for the first time, begin with simple geometric primitives such as points, vectors, segments, and triangles. A strong understanding of these building blocks makes more advanced algorithms significantly easier to understand.
- Focus on geometric predicates before studying large geometric data structures. Orientation tests, barycentric coordinates, and point-classification methods appear repeatedly throughout computational geometry and numerical simulation.
- Implement basic algorithms such as segment intersection, point-in-polygon testing, and convex hull construction. These examples illustrate how simple predicates can be combined to solve more complex geometric problems.
- Pay careful attention to numerical robustness. Many geometric algorithms that appear correct mathematically can fail in practice because of floating-point roundoff and degeneracies.
- After mastering basic geometric algorithms, explore Delaunay triangulations and Voronoi diagrams. These structures provide valuable insight into mesh generation, interpolation, and scientific-computing applications.
- Experiment with spatial-search structures and compare their performance with brute-force approaches. Such comparisons help develop intuition about algorithmic complexity and scalability.
- For Rust implementations, emphasize clear data structures, strong type safety, and consistent classification logic. Reliability is often more important than raw performance when developing geometric software.
- Most importantly, remember that geometric algorithms are often components within larger computational pipelines. Correctness, robustness, and maintainability should therefore be considered alongside efficiency.

## 21.9.3. Further Learning with GenAI

To deepen your understanding of computational geometry and its implementation in Rust, consider exploring the following prompts:

 1. Explain the role of orientation predicates in computational geometry and demonstrate their use in common geometric algorithms.
 2. Implement a robust segment-intersection algorithm in Rust and analyze how degeneracies affect its behavior.
 3. Compare ray-crossing and winding-number methods for point-in-polygon testing and discuss their advantages and limitations.
 4. Implement a convex hull algorithm and compare the complexity and performance of different construction strategies.
 5. Explain the duality between Delaunay triangulations and Voronoi diagrams and visualize their geometric relationships.
 6. Implement a Bowyer–Watson Delaunay triangulation algorithm in Rust and analyze its computational complexity.
 7. Compare uniform grids, k-d trees, octrees, and bounding-volume hierarchies for nearest-neighbor and range-search queries.
 8. Analyze how mesh quality influences interpolation accuracy and numerical simulation stability.
 9. Explain symbolic perturbation and exact-predicate methods for handling degeneracies in geometric algorithms.
10. Design a robust geometric processing pipeline that integrates predicates, intersections, triangulations, and spatial searches for a scientific-computing application.

By exploring these prompts, readers can develop a deeper understanding of both geometric theory and practical implementation strategies.

## 21.9.4. Homework Exercises

To reinforce your understanding of the material covered in this chapter, complete the following exercises:

 1. Implement orientation predicates in two and three dimensions and analyze their behavior for nearly degenerate configurations.
 2. Develop a segment-intersection algorithm and test it on cases involving overlapping, collinear, and nearly parallel segments.
 3. Implement both ray-crossing and winding-number point-in-polygon algorithms and compare their performance and robustness.
 4. Construct a convex hull algorithm for two-dimensional point sets and analyze its computational complexity.
 5. Implement barycentric-coordinate interpolation on triangular elements and evaluate interpolation accuracy for different mesh qualities.
 6. Develop a Delaunay triangulation algorithm and investigate how insertion order influences computational performance.
 7. Compare brute-force nearest-neighbor search with a k-d tree implementation for datasets of increasing size.
 8. Analyze the effects of mesh quality on interpolation and geometric approximation using both regular and distorted meshes.
 9. Implement filtered predicates or exact arithmetic techniques and compare their robustness with standard floating-point implementations.
10. Select a real-world application such as mesh generation, scientific visualization, geographic information systems, particle simulation, or collision detection. Design a geometric processing pipeline and justify the algorithms chosen for each stage.

Computational geometry provides many of the fundamental tools required for modern scientific computing. The algorithms developed in this chapter support simulation, mesh generation, spatial analysis, visualization, optimization, and data processing across a wide range of disciplines. As computational models become larger and more complex, the need for robust and efficient geometric algorithms continues to grow. By mastering the concepts and implementation techniques presented in this chapter, readers will be well prepared to develop reliable geometric software and to apply these methods in advanced scientific and engineering applications.

## References

 1. Bálint, C., Bán, R. and Valasek, G. (2024) ‘Computing the cut locus, Voronoi diagram, and signed distance function of polygons’, *Computer Aided Geometric Design*, 114, 102388. doi: 10.1016/j.cagd.2024.102388.
 2. Bartels, T., Fisikopoulos, V. and Weiser, M. (2023) ‘Fast floating-point filters for robust predicates’, *BIT Numerical Mathematics*, 63, Article 31. doi: 10.1007/s10543-023-00975-x.
 3. Benthin, C., Meister, D., Barczak, J., Mehalwal, R., Tsakok, J. and Kensler, A. (2024) ‘H-PLOC: Hierarchical Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction’, *Proceedings of the ACM on Computer Graphics and Interactive Techniques*, 7(3), 14 pp. doi: 10.1145/3675377.
 4. Chen, H., Ullrich, P.A. and Panetta, J. (2026) ‘Fast and Accurate Intersections on a Sphere’, *SIAM Journal on Scientific Computing*, 48(2), pp. B208–B232. doi: 10.1137/25M1737614.
 5. Dieci, L., Difonzo, F.V. and Sukumar, N. (2024) ‘Nonnegative moment coordinates on finite element geometries’, *Mathematics in Engineering*, 6(1), pp. 81–99. doi: 10.3934/mine.2024004.
 6. Elshakhs, Y.S., Deliparaschos, K.M., Charalambous, T., Oliva, G. and Zolotas, A. (2024) ‘A Comprehensive Survey on Delaunay Triangulation: Applications, Algorithms, and Implementations Over CPUs, GPUs, and FPGAs’, *IEEE Access*, 12, pp. 12562–12585. doi: 10.1109/ACCESS.2024.3354709.
 7. Fuda, C. and Hormann, K. (2024) ‘A new stable method to compute mean value coordinates’, *Computer Aided Geometric Design*, 111, 102310. doi: 10.1016/j.cagd.2024.102310.
 8. Gæde, E.T., Gørtz, I.L., van der Hoog, I., Krogh, C. and Rotenberg, E. (2024) ‘Simple and Robust Dynamic Two-Dimensional Convex Hull’, in *Proceedings of the SIAM Symposium on Algorithm Engineering and Experiments*, pp. 144–156. doi: 10.1137/1.9781611977929.11.
 9. Gao, W. and Chen, R. (2025) ‘Parallel 3D Delaunay Triangulation on the GPU’, *Computer-Aided Design*, 189, 103933. doi: 10.1016/j.cad.2025.103933.
10. Guo, J.-P. and Fu, X.-M. (2024) ‘Exact and Efficient Intersection Resolution for Mesh Arrangements’, *ACM Transactions on Graphics*, 43(6), pp. 1–14. doi: 10.1145/3687925.
11. Ji, Y., Liu, S., Guo, J.-P., Su, J.-P. and Fu, X.-M. (2024) ‘Evolutionary multi-objective high-order tetrahedral mesh optimization’, *Computer Aided Geometric Design*, 111, 102302. doi: 10.1016/j.cagd.2024.102302.
12. Kwon, H., Oh, S. and Baek, J.-W. (2024) ‘Algorithmic Efficiency in Convex Hull Computation: Insights from 2D and 3D Implementations’, *Symmetry*, 16, 1590. doi: 10.3390/sym16121590.
13. Laso, R. and Yermo, M. (2026) ‘Cheesemap: A high-performance point-indexing data structure for neighbor search in LiDAR data’, *Future Generation Computer Systems*, 175, 108060. doi: 10.1016/j.future.2025.108060.
14. Lei, N., Feng, Y., Duan, J., Zhang, P., Zheng, X. and Gu, X. (2024) ‘A Survey of Finite Element Mesh Generation Methods’, *Journal of the University of Electronic Science and Technology of China*, 53(6), pp. 816–843. doi: 10.12178/1001-0548.2024246.
15. Lévy, B. (2025) ‘Exact Predicates, Exact Constructions and Combinatorics for Mesh CSG’, *ACM Transactions on Graphics*, 44(5). doi: 10.1145/3744642.
16. Lévy, B., Ray, N., Mérigot, Q. and Leclerc, H. (2025) ‘Large-scale semi-discrete optimal transport with distributed Voronoi diagrams’, *Journal of Computational Physics*, 542, 114374. doi: 10.1016/j.jcp.2025.114374.
17. López, J. and Hernández, J. (2024) ‘On polytope intersection by half-spaces and hyperplanes for unsplit geometric volume of fluid methods on arbitrary grids’, *Computer Physics Communications*, 300, 109167. doi: 10.1016/j.cpc.2024.109167.
18. Qiu, J., Qi, L., Zhu, X., Tang, M., Liu, Y., Xu, G. and Pang, Y. (2026) ‘MSQ-meshing: a multi-constraint size-adaptive quad-dominant mesh generation method for complex surfaces’, *Computers & Structures*, 327, 108257. doi: 10.1016/j.compstruc.2026.108257.
19. Romano, P.K., Myers, P.A., Johnson, S.R., Kolšek, A. and Shriwise, P.C. (2025) ‘Point containment algorithms for constructive solid geometry with unbounded primitives’, *Computer-Aided Design*, 178, 103803. doi: 10.1016/j.cad.2024.103803.
20. Skala, V. (2025a) ‘A new fully projective O(lg N) line convex polygon intersection algorithm’, *The Visual Computer*, 41, pp. 1241–1249. doi: 10.1007/s00371-024-03413-3.
21. Skala, V. (2025b) ‘A new fully projective O(log N) point-in-convex polygon algorithm: a new strategy’, *The Visual Computer*, 41(7), pp. 4839–4850. doi: 10.1007/s00371-024-03693-9.
22. Teuscher, B., Walther, P., Wang, J. and Werner, M. (2025) ‘A Taxonomy of Point Cloud Search’, *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, X-4/W6-2025, pp. 185–192. doi: 10.5194/isprs-annals-X-4-W6-2025-185-2025.
23. Ting, K.M., Washio, T., Zhu, Y., Xu, Y. and Zhang, K. (2024) ‘Is it possible to find the single nearest neighbor of a query in high dimensions?’, *Artificial Intelligence*, 336, 104206. doi: 10.1016/j.artint.2024.104206.
24. Viñambres, P.D., Yermo, M., Alcaraz, S.R., Lorenzo, O.G., Rivera, F.F. and Cabaleiro, J.C. (2026) ‘Efficient Neighbourhood Search in 3D Point Clouds Through Space-Filling Curves and Linear Octrees’, *arXiv / Computing Research Repository*. arXiv:2603.06771.
25. Zou, X., Lo, S.B., Sevilla, R., Hassan, O. and Morgan, K. (2024) ‘The generation of tetrahedral meshes for NURBS-enhanced FEM’, *Engineering with Computers*, 40, pp. 3949–3977. doi: 10.1007/s00366-024-02004-z.
