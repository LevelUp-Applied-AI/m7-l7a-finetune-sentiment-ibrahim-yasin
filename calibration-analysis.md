# Calibration Analysis

## Reliability diagram interpretation

The reliability diagram shows that the model becomes more accurate as prediction confidence increases. Lower-confidence buckets around the 0.3–0.5 range achieved empirical accuracies around 0.4–0.5, while higher-confidence buckets near 0.8–1.0 achieved accuracies closer to 0.7–0.9.

Several buckets fall below the perfect calibration line, which indicates that the model is over-confident in some confidence ranges. For example, predictions in the 0.7–0.8 confidence bucket achieved noticeably lower empirical accuracy than their predicted confidence. However, the highest-confidence buckets were more aligned with the diagonal and therefore better calibrated.

Overall, the model appears moderately calibrated but still somewhat over-confident in mid-to-high confidence regions.

## Expected Calibration Error

The model produced an Expected Calibration Error (ECE) that indicates a moderate gap between predicted confidence and true accuracy.

This means the model’s probabilities should not be interpreted as perfectly reliable confidence estimates in production settings. While the model predictions are often correct at higher confidence levels, the confidence values themselves are sometimes inflated relative to actual performance.

The calibration quality is acceptable for experimentation, but additional calibration improvements would be beneficial before deployment in a high-trust production environment.

## A specific calibration pattern

One clear pattern is over-confidence in several mid-confidence buckets. The model frequently predicts probabilities around 0.7–0.8 even when the empirical accuracy is lower.

This likely occurred because the DistilBERT classifier was optimized primarily for classification accuracy and macro-F1 during fine-tuning, rather than probability calibration. Cross-entropy training often encourages confident predictions, especially on easier or majority-class examples.

The weaker performance on the neutral class also suggests that class overlap and ambiguous sentiment boundaries contribute to calibration error.

## A proposed engineering action

A practical production improvement would be temperature scaling to better calibrate prediction probabilities without retraining the full model.

Another useful action would be threshold-based abstention. For example, predictions below a confidence threshold (such as 0.6) could be flagged for human review instead of being automatically trusted.

Additionally, collecting more difficult neutral-sentiment examples could help reduce confusion between classes and improve calibration consistency across confidence buckets.