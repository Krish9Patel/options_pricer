# LSMC AAD Options Pricer

This project is an options pricing engine that uses the **Least Squares Monte Carlo (LSMC)** method to price American-style options, paired with **Adjoint Algorithmic Differentiation (AAD)** to compute exact sensitivities (Greeks) efficiently.

The focus is on production-grade systems engineering rather than another textbook LSMC implementation: a flat, cache-aligned tape with no heap allocation per node; a two-pass (RNG checkpointing) architecture that keeps the $O(N^3)$ regression off the computation graph; and sigmoid smoothing of the exercise boundary to produce stable second-order Greeks. The goal is performance and Greek quality comparable to what's described in Matlogica's published benchmarks.

This started as a fun way to explore how the math behind financial markets translates into software; I wanted to see what that intersection looked like in practice and push myself to learn C++ while building a cool project. I hope to eventually build a full-stack financial simulation library in the future.

> **Status:** Active refactor in progress. The repository currently contains a working academic-grade LSMC + AAD implementation; the production architecture described below is being implemented in phases.

## Architecture

The pricing engine is built on a **Two-Pass (RNG Checkpointing) Architecture** to minimize tape overhead and maximize performance:

1. **Pass 1 (Calibration):**
   - Runs in standard `double` precision, without recording to the computation graph.
   - Simulates geometric Brownian motion paths using a fixed RNG seed.
   - Performs backward induction and cross-sectional regressions to learn the exercise boundary (the continuation value).
   - Stores the regression coefficients ( $\beta_t$ ) for each timestep.

2. **Pass 2 (Valuation & Greeks):**
   - Re-runs the exact same paths using `AD_double` (the custom automatic differentiation type).
   - Uses the identical RNG seed to recreate the exact trajectories.
   - Evaluates the exercise logic using the pre-computed $\beta_t$ coefficients as constants — so the expensive linear algebra never touches the tape.
   - Computes the option price and backpropagates through the flattened tape to extract Delta, Vega, Rho, and Theta in a single backward sweep.

## Mathematical Foundation

### 1. LSMC Backward Induction

The Longstaff-Schwartz algorithm determines the optimal exercise strategy by comparing the immediate exercise value to the expected continuation value at each timestep. The continuation value is estimated via cross-sectional regression on the in-the-money paths, evaluated at the current spot $S_t$ for each path, using a polynomial basis:

$$\text{Continuation Value}(S_t) \approx \beta_0 + \beta_1 S_t + \beta_2 S_t^2 + \cdots$$

### 2. Adjoint Algorithmic Differentiation (AAD)

AAD (also known as reverse-mode automatic differentiation, or backpropagation in the ML literature) computes derivatives by applying the chain rule in reverse over a recorded computation graph. Given an option price $V(S_0, \sigma, r, T)$, AAD calculates all first-order sensitivities (Delta, Vega, Rho, Theta) at a computational cost bounded by a small constant multiple of the price computation itself — this is the *cheap gradient principle* (typically 3–4× the forward pass for reverse-mode AD). This stands in stark contrast to finite differences, which require one full re-pricing per Greek.

### 3. Sigmoid Smoothing

A hard exercise boundary uses a step function. At each timestep, the cash flow is set to the exercise value if exercising is optimal, and otherwise rolled forward as the discounted future cash flow:

$$\text{Cash Flow} = \text{Exercise} \cdot \mathbb{1}[\text{Exercise} > \text{Continuation}] + \text{Future} \cdot \mathbb{1}[\text{Exercise} \le \text{Continuation}]$$

The derivative of this indicator function is the Dirac delta, meaning gradients evaluate to zero almost everywhere. This causes AAD to return zero or extremely noisy Greeks for American options — especially Gamma, which depends on the derivative of an already-discontinuous Delta.

To solve this, we replace the hard decision with a numerically stable **sigmoid activation function** that produces a probability of exercise:

$$P(\text{exercise}) = \sigma\left(\frac{\text{Exercise} - \text{Continuation}}{\epsilon}\right) = \frac{1}{1 + \exp\left(-\frac{\text{Exercise} - \text{Continuation}}{\epsilon}\right)}$$

The cash flow then becomes a continuous, differentiable weighted average:

$$\text{Cash Flow} = P \cdot \text{Exercise} + (1 - P) \cdot \text{Future}$$

This ensures stable gradients across the entire computation graph at the cost of a small $O(\epsilon)$ price bias, which is typically well below Monte Carlo variance for $\epsilon \approx 0.01 \cdot K$.

## References & Inspiration

The refactored architecture draws on the following work:

- Longstaff, F. A. & Schwartz, E. S. (2001). *Valuing American Options by Simulation: A Simple Least-Squares Approach.* Review of Financial Studies, 14(1), 113–147. — The foundational LSMC paper.
- Savine, A. (2018). *Modern Computational Finance: AAD and Parallel Simulations.* Wiley. — Comprehensive treatment of AAD, tape design, and parallel Monte Carlo for derivatives pricing.
- Capriotti, L., Jiang, Y. & Macrina, A. (2017). *AAD and least-square Monte Carlo: Fast Bermudan-style options and XVA Greeks.* — Detailed treatment of AAD applied specifically to regression-based Monte Carlo.
- Gulbinsov, D. (Matlogica). *Applying AAD to American Monte Carlo Option Pricing.* — Pathwise AAD architecture for production-grade risk engines.
- Griewank, A. & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation.* SIAM. — Standard reference on AAD theory and the cheap gradient principle.