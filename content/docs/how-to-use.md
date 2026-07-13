---
weight: 200
title: "How to Use This Book"
description: "Feynman's way of learning"
icon: "school"
date: "2026-07-06T00:00:00+07:00"
lastmod: "2026-07-06T00:00:00+07:00"
katex: true
draft: false
toc: true
---

{{% alert icon="💡" context="info" %}}
<strong>"<em>If you want to learn something really well, teach it to someone else.</em>" — Richard Feynman</strong>
{{% /alert %}}

<div class="container my-5 p-4" style="background-color: var(--bg-color); color: var(--text-color);">
  <style>
    :root:not([data-dark-mode]) {
      --bg-color: #ffffff;
      --text-color: #000000;
      --accordion-bg: #f8f9fa;
      --accordion-text: #000000;
      --accordion-border: #ddd;
      --alert-bg: #e9ecef;
      --alert-text: #000000;
      --alert-border: #ccc;
    }
    :root[data-dark-mode] {
      --bg-color: #121212;
      --text-color: #e0e0e0;
      --accordion-bg: #1e1e1e;
      --accordion-text: #e0e0e0;
      --accordion-border: #333;
      --alert-bg: #333;
      --alert-text: #e0e0e0;
      --alert-border: #444;
    }
    .accordion-item.custom {
      background-color: var(--accordion-bg);
      color: var(--accordion-text);
      border: 1px solid var(--accordion-border);
    }
    .accordion-button.custom {
      background-color: var(--accordion-bg);
      color: var(--accordion-text);
      border: 1px solid var(--accordion-border);
    }
    .accordion-button.custom:not(.collapsed) {
      background-color: var(--accordion-border);
      color: #fff;
    }
    .accordion-body.custom {
      background-color: var(--accordion-bg);
      color: var(--accordion-text);
    }
    .list-group-item.custom {
      background-color: var(--accordion-bg);
      color: var(--accordion-text);
      border: 1px solid var(--accordion-border);
    }
    .alert.custom {
      background-color: var(--alert-bg);
      color: var(--alert-text);
      border-color: var(--alert-border);
    }
  </style>
  <div class="mb-4">
    <p class="text-justify">
      To use the <em>NCVR - Numerical Computing via Rust</em> book effectively, embrace a Richard Feynman-inspired approach—one that emphasizes deep understanding, curiosity, and hands-on experimentation. Whether you follow a structured, sequential approach or dive straight into topics that interest you, this book is designed to support your unique learning journey. Explore, experiment, and let the rich content guide you to master numerical computing and Rust.
    </p>
  </div>
  <div class="accordion" id="howToUseAccordion">
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingOne">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseOne" aria-expanded="false" aria-controls="collapseOne">
          Embrace the Curiosity
        </button>
      </h2>
      <div id="collapseOne" class="accordion-collapse collapse" aria-labelledby="headingOne" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Understand the Foundations:</strong> Begin with Part I to get a solid grasp of Rust and its application to numerical computing. Dive into each concept with Feynman’s curiosity. Ask probing questions using GenAI, clarify fundamental principles, and generate simple code examples. Learn <em>why</em> Rust is effective for scientific computing—not just <em>how</em> to use it.
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingTwo">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
          Master Linear Systems and Function Approximation
        </button>
      </h2>
      <div id="collapseTwo" class="accordion-collapse collapse" aria-labelledby="headingTwo" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Explore Core Algorithms:</strong> In Part II, delve into the numerical solution of linear systems, interpolation, quadrature, and the evaluation of functions. Break each method down into its core components. Use GenAI to generate hands-on exercises and practical examples with crates such as <code>nalgebra</code> and <code>ndarray</code>.
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingThree">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseThree" aria-expanded="false" aria-controls="collapseThree">
          Reason about Randomness and Data Ordering
        </button>
      </h2>
      <div id="collapseThree" class="accordion-collapse collapse" aria-labelledby="headingThree" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Work with Random Numbers and Sorting:</strong> In Part III, study random number generation, sorting, and selection. Implement and test these building blocks using crates like <code>rand</code>, and use GenAI to reason about statistical quality and algorithmic complexity.
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingFour">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseFour" aria-expanded="false" aria-controls="collapseFour">
          Engage with Optimization, Roots, and Eigensystems
        </button>
      </h2>
      <div id="collapseFour" class="accordion-collapse collapse" aria-labelledby="headingFour" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Solve and Optimize:</strong> In Part IV, tackle root finding, minimization and maximization of functions, and eigensystems with a hands-on mindset. Use GenAI to break complex nonlinear problems into manageable parts and to validate convergence behavior.
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingFive">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseFive" aria-expanded="false" aria-controls="collapseFive">
          Explore Spectral Methods with Enthusiasm
        </button>
      </h2>
      <div id="collapseFive" class="accordion-collapse collapse" aria-labelledby="headingFive" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Investigate the Frequency Domain:</strong> In Part V, adopt Feynman’s experimental spirit to explore the Fast Fourier Transform and its many applications. Use crates like <code>rustfft</code> to analyze signals, and leverage GenAI to deepen your understanding and troubleshoot transforms.
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingSix">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseSix" aria-expanded="false" aria-controls="collapseSix">
          Model Data and Draw Inferences
        </button>
      </h2>
      <div id="collapseSix" class="accordion-collapse collapse" aria-labelledby="headingSix" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Experiment with Statistics and Modeling:</strong> In Part VI, describe data statistically, fit models, and perform classification and inference using crates like <code>ndarray</code> and <code>statrs</code>. Use GenAI for scenario exploration and iterative learning.
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingSeven">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseSeven" aria-expanded="false" aria-controls="collapseSeven">
          Solve Differential and Integral Equations Actively
        </button>
      </h2>
      <div id="collapseSeven" class="accordion-collapse collapse" aria-labelledby="headingSeven" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Integrate Equations:</strong> In Part VII, solve ordinary and partial differential equations, two-point boundary value problems, and integral equations. Experiment with different schemes using crates like <code>nalgebra</code> and <code>ndarray</code>, and use GenAI to analyze stability and accuracy.
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingEight">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseEight" aria-expanded="false" aria-controls="collapseEight">
          Delve into Geometry and Trustworthy Execution
        </button>
      </h2>
      <div id="collapseEight" class="accordion-collapse collapse" aria-labelledby="headingEight" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Investigate Geometry and Reliability:</strong> In Part VIII, explore computational geometry and the infrastructure for trustworthy numerical execution—reproducibility, testing, and verification. Validate your understanding with interactive exercises and GenAI-assisted review.
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingNine">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseNine" aria-expanded="false" aria-controls="collapseNine">
          Leverage GenAI for Deeper Insights
        </button>
      </h2>
      <div id="collapseNine" class="accordion-collapse collapse" aria-labelledby="headingNine" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Generate and Refine Understanding:</strong> Throughout the book, use GenAI to break down complex topics, offer alternative explanations, and generate new insights. Let it be your tool for continuous inquiry and learning.
            </li>
          </ul>
        </div>
      </div>
    </div>
    <div class="accordion-item custom">
      <h2 class="accordion-header" id="headingTen">
        <button class="accordion-button collapsed custom" type="button" data-bs-toggle="collapse" data-bs-target="#collapseTen" aria-expanded="false" aria-controls="collapseTen">
          Join the RantAI Academy
        </button>
      </h2>
      <div id="collapseTen" class="accordion-collapse collapse" aria-labelledby="headingTen" data-bs-parent="#howToUseAccordion">
        <div class="accordion-body custom">
          <ul class="list-group">
            <li class="list-group-item custom">
              <strong>Engage and Share:</strong> Participate in RantAI Academy’s Telegram or Discord channels to share insights and solutions. Use GenAI and community-driven knowledge to further your understanding and collaborative learning.
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
  <div class="mt-4">
    <p class="text-justify">
      By integrating Feynman’s learning philosophy with interactive GenAI tools, you will not only gain a deep understanding of numerical computing but also enjoy a more engaging and effective learning experience with the NCVR book.
    </p>
  </div>
</div>
