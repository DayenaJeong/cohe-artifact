# COHE Claim-Tier Guide

COHE tiers are reporting categories. They bind wording to the evidence level observed for a specified proxy, target, domain, and action protocol.

| Tier | Required evidence pattern | Permitted wording | Forbidden overclaim | COHE example |
|---|---|---|---|---|
| Validated task surrogate | Dependence, held-out prediction, and operational transfer are consistently positive for the stated target/domain. | "Reliable surrogate for this target/domain under this protocol." | Claiming general validity outside the audited target/domain. | Not assigned to the audited CE/margin generative proxies. |
| Weak surrogate evidence | Small nonzero dependence or R2; operational gains are absent, limited, or inconsistent. | "Weak evidence; validate before use." | Treating a small correlation or R2 as enough for filtering or pruning. | CE/margin evidence for several label-free proxies remains weak. |
| Target-specific relation | Signal appears for one target, learner, or dynamics definition but does not generalize across targets. | "Target-specific association, not a unified hardness scalar." | Rebranding a dynamics or sensitivity signal as CE/margin hardness. | DINOv2 first-learning; GradNorm sensitivity associations. |
| No reliable transfer | Dependence is weak, held-out R2 is near zero, or downstream gain is unreliable. | "Not a reliable surrogate in this evaluated setting." | Calling the proxy a CE/margin surrogate without task-wise validation. | Gen-Hard for CE/margin pruning/filtering in the evaluated protocol. |
| Operational-only effect | Downstream gain appears without dependence or predictive-validity support. | "Useful policy in this protocol; not scalar-surrogate evidence." | Claiming the scalar measures discriminative difficulty. | A hypothetical transfer-only gain would remain operational-only until dependence/prediction support is shown. |

When tiers differ across targets, report target-specific tiers rather than collapsing them into one universal "difficulty" label.
