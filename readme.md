F(F)CP : Predictive Inference with (Fast) Feature Conformal Prediction
======================

![FFCP Logo](./logo.png)

Conformal prediction (CP) is a distribution-free framework for uncertainty quantification
that provides valid prediction intervals with finite-sample coverage guarantees. Tradition-
ally, CP operates in the output space, relying on residual-based non-conformity scores.
However, this paradigm fails to utilize the rich features learned by modern deep learn-
ing models, potentially leading to information loss and suboptimal performance. In this
paper, we extend the scope of conformal prediction to semantic feature spaces in deep
representation learning. We first propose Feature Conformal Prediction (FCP), detailing
how to adapt CP techniques within the feature space and then convert these feature-
space predictions back to the output space. Furthermore, recognizing that the computa-
tional costs of FCP limit its practicality in large-scale settings, we propose Fast Feature
Conformal Prediction (FFCP) which accelerates FCP by employing a first-order Taylor
approximation to linearize the feature-to-output mapping. Experimental results validate
that (F)FCP produces tighter prediction bands compared to standard CP techniques. We
further demonstrate that the feature-level framework can be extended to other variants
of CP, including CQR, Localized CP and RAPS. We also provide several applications of
(F)FCP, including image segmentation and large language models


* Conformalized Quantile Regression (CQR) : [Paper](https://arxiv.org/pdf/1905.03222) & [Code](https://github.com/yromano/cqr?utm_source=catalyzex.com)

*  Regularized Adaptive Prediction Sets (RAPS) : [Paper](https://arxiv.org/abs/2009.14193) & [Code](https://github.com/aangelopoulos/conformal_classification/tree/master)

* Localized Conformal Prediction (LCP) : [Paper](https://arxiv.org/pdf/2106.08460)

* LLMs via Uncertainty Quantification : [Paper](https://arxiv.org/pdf/2401.12794)