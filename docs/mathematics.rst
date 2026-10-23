Mathematical and numerical conventions
======================================

A fitted model combines correlation, covariance and fourth central
co-moments for a current return window. CV obtains dependence coefficients
from random splits of that window; historical AO and HOAO use coefficients
calibrated on earlier window pairs. In both cases, the current window
supplies marginal scales and the bases in which the coefficients are applied.

Observations and standardization
--------------------------------

Let :math:`\mathbf X\in\mathbb R^{\Delta t\times n}` contain
:math:`\Delta t` observations of :math:`n` assets. Rows index time and
columns index assets. The current sample mean is
:math:`\hat{\boldsymbol\mu}\in\mathbb R^n`; marginal sample standard
deviations with population normalization (``ddof=0``) form
:math:`\widehat{\mathbf D}=\operatorname{diag}(\hat{\boldsymbol\sigma})
\in\mathbb R^{n\times n}`. Standardized observations are

.. math::

   Z_{ti}=\frac{X_{ti}-\hat\mu_i}{\hat\sigma_i},\qquad
   \hat\sigma_i^2=\frac{1}{\Delta t}
   \sum_{t=1}^{\Delta t}(X_{ti}-\hat\mu_i)^2.

The original fitting window determines these scales. Covariance filtering
then operates on the dependence structure, rather than carrying historical
volatility levels into the current estimate.

CV standardizes the full fitting window before splitting and re-centers each
training and testing subsample. Historical AO and HOAO standardize the
training and testing windows independently. Changing this convention changes
the estimator.

Correlation and covariance
--------------------------

For each training sample, obtain a covariance eigenbasis
:math:`\mathbf U_2\in\mathbb R^{n\times n}` with modes ordered by
decreasing eigenvalue. Project testing observations into that basis and
average their squared coordinates. Averaging across splits or historical
pairs gives :math:`\mathbf a\in\mathbb R^n`.

In the current fitting basis, first reconstruct
:math:`\widetilde{\mathbf C}=\mathbf U_2\operatorname{diag}(\mathbf a)
\mathbf U_2^{\mathsf T}`. Normalize by its diagonal before restoring the
current marginal scales:

.. math::

   \widehat C_{ij}=
   \frac{\widetilde C_{ij}}
   {\sqrt{\widetilde C_{ii}\widetilde C_{jj}}},\qquad
   \widehat{\boldsymbol\Sigma}=
   \widehat{\mathbf D}\widehat{\mathbf C}\widehat{\mathbf D}.

This congruence normalization preserves positive semidefiniteness and makes
the diagonal of the filtered correlation equal to one. Merely overwriting
the diagonal would be a different operation.

Fourth-order basis without a dense tensor
-----------------------------------------

The fourth-order tensor
:math:`\widehat{\mathcal K}_z\in\mathbb R^{n\times n\times n\times n}`
is the sample central co-moment of standardized observations. Its mode-one
unfolding has shape :math:`n\times n^3`, with row index :math:`i` and merged
column indices :math:`(j,k,l)` in C order (:math:`l` varies fastest).
The left singular vectors supply a basis
:math:`\mathbf U_4\in\mathbb R^{n\times n}`, distinct from the covariance
eigenbasis.

The basis can be computed from a square Gram matrix without forming the
fourth-order tensor:

.. math::

   \mathbf G=\frac{1}{\Delta t^2}\mathbf Z^{\mathsf T}
      [ (\mathbf Z\mathbf Z^{\mathsf T})^{\circ3} ]\mathbf Z.

Here :math:`\circ3` is an elementwise cube, not a matrix cube. The
implementation blocks observation products to limit temporary memory.
The time cost is
:math:`O(\Delta t^2n+\Delta t n^2+n^3)`, including diagonalization,
rather than constructing all :math:`n^4` tensor entries.

Even cores and portfolio contractions
-------------------------------------

Given centered testing observations projected into the training basis,
:math:`\mathbf Y\in\mathbb R^{\Delta t_{\rm test}\times n}`, only even
index multiplicities survive the sign-invariant projection. The core is
therefore represented by
:math:`\mathbf B\in\mathbb R^{n\times n}`:

.. math::

   B_{ij}=\frac{1}{\Delta t_{\rm test}}
          \sum_t Y_{ti}^2Y_{tj}^2.

CV averages these matrices over random splits; HOAO averages them over
historical chronological pairs. Sample-even uses the same current window
for the basis and the coefficients. The full core keeps entries of the
form :math:`(i,i,i,i)` and permutations of :math:`(i,i,j,j)`.

For portfolio weights :math:`\mathbf w\in\mathbb R^n`, define
:math:`\mathbf v=\mathbf U_4^{\mathsf T}\widehat{\mathbf D}\mathbf w`.
The fourth central moment in return units is

.. math::

   \hat m_4(\mathbf w)
   =3(\mathbf v\circ\mathbf v)^{\mathsf T}\mathbf B
       (\mathbf v\circ\mathbf v)
    -2\sum_i B_{ii}v_i^4.

This uses :math:`O(n^2)` arithmetic and storage. The portfolio variance and
Pearson kurtosis are

.. math::

   \hat\sigma_p^2=\mathbf w^{\mathsf T}
     \widehat{\boldsymbol\Sigma}\mathbf w,
   \qquad
   \hat\kappa_p=\frac{\hat m_4(\mathbf w)}{\hat\sigma_p^4}.

The sample estimator evaluates the fourth power of centered portfolio
returns directly, retaining the unfiltered moment without constructing a
dense tensor. Filtered estimators use the compact expression above.

Optimization
------------

Let :math:`s^2` be the sample variance of the equal-weight portfolio in
the original fitting window, before filtering. The allocation objective is

.. math::

   \min_{\mathbf w:\,\mathbf 1^{\mathsf T}\mathbf w=1}
   \frac{\mathbf w^{\mathsf T}\widehat{\boldsymbol\Sigma}\mathbf w}{s^2}
   +\gamma\frac{\hat m_4(\mathbf w)}{s^4},\qquad\gamma\geq0.

The budget is the only constraint. The fourth-moment term is convex for
cores constructed from projected empirical observations and their averages.
With a covariance that is positive definite on the budget tangent space,
the objective has a unique optimum. Zero penalty yields minimum variance.
Analytic derivatives avoid finite-difference tensor evaluations.

The fourth moment's gradient in rotated coordinates is

.. math::

   \nabla_{\mathbf v}\hat m_4
   =12\mathbf v\circ[\mathbf B(\mathbf v\circ\mathbf v)]
    -8\operatorname{diag}(\mathbf B)\circ\mathbf v^{\circ3}.

Multiplication by :math:`\widehat{\mathbf D}\mathbf U_4` maps this to
weight coordinates. Hessian-vector products use the same compact core.
The returned diagnostics include the budget-constrained gradient residual.
Acceptance requires a budget error no larger than ``1e-10`` and a residual
no larger than ``max(1e-6, 10 * tol)``. An unsuccessful solve raises
``OptimizationError`` with the diagnostic candidate attached.

Limitations and numerical reproducibility
-----------------------------------------

Second and fourth moments are filtered separately. They need not be jointly
realizable by one distribution, so their ratio can yield kurtosis below one.
The package reports that result without clipping. The compatible variants
discussed in the paper's supplementary information are not implemented.

The even projection is invariant to signs of basis vectors but is generally
not invariant to rotations inside a degenerate eigenspace. Numerically tied
eigenvalues use a deterministic coordinate-projector basis. Rank-deficient
windows are handled through an SVD row-space decomposition and a canonical
null-space completion, including when there are more assets than observations.
This avoids dense fourth-order fallback allocations, but can differ from
reference implementations that let a dense eigensolver choose the tied basis.
Calibration and fitting use the same convention. Floating-point changes
between BLAS libraries or devices can still affect such cases;
compare invariant moments and documented tolerances rather than expecting
bitwise-identical basis columns. Double precision is used for numerical work.

``to_dense`` is an optional export, not the storage model. A float64 fourth
tensor alone requires :math:`8n^4` bytes; a dense reconstruction may need
additional working storage. The caller-supplied byte budget prevents an
accidental large tensor allocation.
