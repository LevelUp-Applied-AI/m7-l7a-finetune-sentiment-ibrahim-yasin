# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category | Correct | Total | Accuracy |
|---|---|---|---|
| negation | 4 | 6 | 0.67 |
| lexical_trigger | 1 | 5 | 0.20 |
| domain_shift | 3 | 5 | 0.60 |
| length_extreme | 3 | 5 | 0.60 |
| sarcasm | 0 | 5 | 0.00 |
| other | 1 | 2 | 0.50 |

Overall accuracy: 12 / 28 = 0.43

## Confirmed hypotheses

The model struggled most on sarcasm and lexical trigger examples, which confirmed my original hypothesis that the classifier relies heavily on strong sentiment cue words instead of deeper contextual meaning.

For sarcasm, row 21 (“Amazing, another update that somehow breaks everything again.”) was predicted as positive with high confidence even though the intended sentiment was negative. Similarly, row 24 (“Wonderful job deleting my files without any warning.”) was also predicted as positive because the model focused on positive surface phrases like “Wonderful job” instead of the negative outcome.

The lexical trigger category also exposed this weakness. In row 9 (“Excellent customer support… if they ever decide to reply.”), the model predicted positive because of the phrase “Excellent customer support” while ignoring the contradictory clause afterward. Row 10 (“Everything sounds perfect until you actually try using it.”) showed the same pattern.

The model also had difficulty with some negation cases. In row 30 (“The app wasn’t actually that bad after the recent fix.”), the intended meaning was positive, but the model predicted neutral, suggesting it struggled to correctly interpret negated negative phrases.

## Refuted hypotheses

The model handled some categories better than I originally expected. I thought domain-shift examples would fail more consistently, but several neutral non-review sentences were classified correctly. For example, row 13 (“Heavy rain caused several flight delays this morning.”) and row 14 (“Researchers reported a small rise in ocean temperatures this year.”) were both correctly predicted as neutral.

The model also performed surprisingly well on short length-extreme examples. Rows 16 (“Absolutely terrible.”), 17 (“Pretty solid overall.”), and 18 (“Worst update yet.”) were all classified correctly with high confidence. This suggests the classifier learned strong sentiment representations for compact review-style language.

Additionally, some negation examples worked correctly. Rows 1 and 29 were correctly classified as negative, showing that the model can detect straightforward negation patterns when the sentence structure is simple.

## What the results reveal about the decision boundary

The adversarial results suggest that the model’s decision boundary is strongly influenced by high-frequency sentiment cue words such as “amazing,” “perfect,” “excellent,” and “wonderful,” even when later context reverses the polarity. The classifier appears to overweight local lexical signals rather than fully modeling sentence-level meaning.

The results also show that sarcasm is particularly difficult because the literal wording is positive while the intended sentiment is negative. Since the model was fine-tuned mainly on direct review language, it struggles when sentiment depends on pragmatic interpretation rather than explicit wording.

The model additionally showed reduced confidence on long negative examples, often predicting neutral instead. This suggests that longer sequences with mixed clauses weaken the influence of strong negative signals and make the classifier less certain overall.

Finally, the model generalized moderately well to neutral factual language outside the review domain, but sports and recipe examples occasionally drifted toward positive predictions, indicating that unfamiliar non-review contexts can still bias the classifier toward emotional interpretations.