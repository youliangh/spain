# Llama numerical semantics and approximation bounds

The model uses MPFR with 128-bit significands to construct its witness. The
nonlinear circuit represents a rational approximation to exponential, rather
than an exact exponential or an IEEE floating-point execution trace. The bound
below is for that rational function in exact real arithmetic. Spain separately
checks the residual of the quantized circuit.

## Negative exponential

For `x` in `[-128, 0]`, [numeric.rs](src/numeric.rs) evaluates

\[
z=x/1024,\qquad
R(z)=\frac{120+60z+12z^2+z^3}{120-60z+12z^2-z^3},\qquad
E(x)=R(z)^{1024}.
\]

The last power is ten successive squarings. Put `t = -z`, so
`0 <= t <= 1/8`, and define

\[
P(t)=120-60t+12t^2-t^3,\qquad Q(t)=120+60t+12t^2+t^3.
\]

There is no pole or sign ambiguity on this domain: `Q(t) >= 120` and
`P(t) >= 120 - 60/8 - 1/512 > 112`. In particular, `R(-t) > 0`.
Integration by parts gives the exact remainder identity

\[
Q(t)e^{-t}-P(t)
=\frac16\int_0^t e^{-u}u^3(t-u)^3\,du.
\]

Since `e^-u <= 1` and the polynomial integral is `t^7/140`,

\[
0\le e^{-t}-R(-t)\le\frac{t^7}{100800}.
\]

Write `R(-t) = e^-t (1-a)`. Using
`e^t <= 1/(1-t) <= 8/7` on this interval yields

\[
0\le a\le\frac{8t^7}{7\cdot100800}.
\]

Bernoulli's inequality, `(1-a)^1024 >= 1-1024a`, therefore gives

\[
0\le 1-\frac{E(x)}{e^x}
\le\alpha:=\frac{1}{180633600}
\approx5.536069\times10^{-9}<2^{-27}.
\]

Because `e^x <= 1`, the same `alpha` also bounds the absolute error of `E(x)`.
This is stronger than an absolute-error target of `2^-20`. The implementation
rejects inputs outside `[-128, 0]`; the claim does not extend beyond that domain.
The analytic bound applies over the entire interval, independently of tests.
The numerical unit test additionally samples 2,049 points; the observed relative
error at `x = -128` is approximately `4.847004e-9`.

## Stable SiLU and causal softmax

SiLU uses `u = abs(x)`, `e = exp(-u)` and `r = E(-u)`. Its approximate sigmoid
is `1/(1+r)` for `x >= 0`, and `r/(1+r)` for `x < 0`. Thus every denominator
is at least one, and positive exponentials are never formed. On
`x in [-128, 128]`, writing `r=e(1-eta)` with `0 <= eta <= alpha` gives

\[
|\widetilde{\operatorname{SiLU}}(x)-\operatorname{SiLU}(x)|
=\frac{u e\eta}{(1+e)(1+r)}
\le\alpha u e^{-u}
\le\alpha\exp(-1)
<2.037\times10^{-9}<2^{-27}.
\]

For each causal softmax row, only positions up to the current token participate.
Future probabilities are constrained to zero. The maximum is subtracted, and
each allowed score difference must lie in `[-128, 0]`. At least one difference is
zero, so its exponential is exactly one in the real rational function, ensuring
a normalization denominator of at least one.

For exact probabilities `p_i`, write the approximate exponentials as
`e_i(1-eta_i)` and let `bar_eta = sum_i p_i eta_i`. Then

\[
\widetilde p_i=p_i\frac{1-\eta_i}{1-\bar\eta},\qquad
|\widetilde p_i-p_i|\le p_i\frac{\alpha}{1-\alpha}.
\]

Consequently both the maximum component error and the sum of absolute row
errors are at most `alpha/(1-alpha) < 5.536070e-9 < 2^-27`.
These bounds compare one operation on identical inputs. They do not establish
a global logit-error bound after composition through all decoder layers and
weight matrices. That comparison is measured separately against the reference
model, including every decoder layer and the full final logit vector.

## Constraint residual and fixed-point rounding

The default fixed-point denominator is `D = 2^75`. Each finite coefficient and
witness value is rounded to the nearest integer after multiplication by `D`.
The rounding error of one represented real number is at most
`1/(2D) = 2^-76`, approximately `1.323489e-23`. The code rejects nonfinite values
and integer overflow. MPFR execution rounding, coefficient rounding, witness
rounding and challenge projection rounding all contribute to the residual;
they are separate from the exact-real approximation bounds above.

For integer matrices `A_D, B_D, C_D` and integer witness `z_D`, the checked
integer squared residual is

\[
J=\sum_i\left[(A_Dz_D)_i(B_Dz_D)_i-D^2(C_Dz_D)_i\right]^2.
\]

The accepted real residual norm is `sqrt(J)/D^4 <= epsilon`, with default
`epsilon = 2^-40`, approximately `9.094947e-13`. Equivalently, `J/D^8` is
compared with `epsilon^2`. This is an algebraic residual tolerance, **not** a
bound on the difference between model logits, a bound on exponential
approximation, or a guarantee of exact floating-point semantics.

Square witnesses enforce nonnegativity in the ideal real constraints. Under
the approximate relation, `s^2 = y` means that `y` may be slightly negative
within the constraint's allowed residual, and bit constraints similarly mean
proximity to a bit. Domain constraints therefore have a tolerance at their
endpoints. Native witness generation checks the domain directly and rejects
out-of-domain inputs, but the pointwise analytic bounds must not be attributed
unchanged to every arbitrary witness accepted by an approximate circuit.

Very small positive exponentials can round to zero at scale `D`; this is
expected quantization, rather than exact positivity of every encoded value.
For genuine witness generation, the softmax row maximum supplies an exponential
of one, and the stable SiLU denominator contains an explicit one. They therefore
avoid division by a rounded tiny exponential. Relating an arbitrary accepted
approximate witness to ideal model outputs additionally requires propagation
and conditioning bounds; this implementation reports measured model alignment
instead of claiming such a theorem.

## Reading the reports

Keep the following quantities distinct when interpreting the artifacts:

| Quantity | Meaning |
| --- | --- |
| `constraint_squared_error_integer`, `scale_bits`, `epsilon` | The checked numerical circuit residual `sqrt(J)/2^(4*scale_bits)`. |
| Exponential bound `1/180633600` | Uniform exact-real relative and absolute approximation bound on `[-128,0]`. |
| SiLU bound `alpha*exp(-1)` and softmax bound `alpha/(1-alpha)` | Exact-real pointwise/row bounds on the specified domains. |
| Reference layer/logit errors and greedy agreement | Empirical alignment of the complete implemented model with HF, including accumulated approximation and execution differences. |

Successful Spain verification records acceptance of the specified approximate
algebraic relation by the implemented protocol. The inherited benchmark RSA
parameters and security limitations are described in [README.md](README.md).
Meaningful generated text and reference agreement are additional end-to-end
acceptance checks.
