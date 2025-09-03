# Pulmonary Arterial Obstruction Scores

The Qanadli and Mastora scores propose different measures of arterial obstruction caused by pulmonary embolism.
Arterial obstruction, and these scores, are hypothesized to be an indicator of pulmonary embolism severity and/or patient outcome.

Both scores traverse the vascular tree to identify arteries of interest and measure their local obstruction, and then apply a formula to compute a global measure of obstruction based on a (weighted) average of the obstructions of these arteries.

&#160;

## Qanadli Score

The Qanadli score proposes to categorize obstructions in branches in [degrees from 0 to 2](#obstruction-to-degree-conversion),
and then perform a [weighted average](#weighting-obstruction-sites) of these degrees to get a [global score](#score-formula).

### Artery Selection

The score looks at arteries at three anatomical levels: mediastinal, lobar, and segmental.

### Weighting Obstruction Sites

When no obstructions are found, the score follows the vascular tree down to the segmental arteries.
It assigns weights of 1 to segmental arteries ($w_i = 1$), to weight each segmental artery's obstruction equally in the final score.

However, when a thrombus is found in the vascular tree that causes at least a partial obstruction, the branches downstream from the obstruction site are not evaluated.
Rather, the obstruction is weighted by the number of downstream segmental arteries ($w_i$) affected by the obstruction of artery $i$.
This means that obstructions in more central arteries will have a higher weight in the final score, proportional to the number of segmental arteries that are affected.

### Obstruction-to-Degree Conversion

Each artery's obstruction value ($o_i$) is converted to a degree ($d_i$) from 0 to 2.
The original paper only describes the degrees as corresponding to partial obstruction (1) or total obstruction (2).
Therefore, we propose two thresholds, $T\_{partial} = 0.25$ and $T\_{total} = 0.75$ to quantify the degrees used in the score:

- $d_i = 0$ if $o_i < T\_{partial}$
- $d_i = 1$ if $T\_{partial} \\le o_i < T\_{total}$
- $d_i = 2$ if $T\_{total} \\le o_i$

### Score Formula

The final score is the sum of the weighted degrees divided by the maximum possible score (i.e. all arteries obstructed at degree 2), normalizing the result to a [0,1] scale.

$$ \\text{Qanadli score} = \\sum\_{i \\in A} \\frac{ w_i \\cdot d_i}{2 \\cdot w_i} $$

&#160;

## Mastora Score

The Mastora score proposes to categorize obstructions in each artery in [degrees from 0 to 5](#obstruction-to-degree-conversion-1),
and then average these degrees across selected arteries to get a [global score](#score-formula-1).
[Three sets of arteries can be selected](#artery-selection-1) for the global score, based on their anatomical level.

### Artery Selection

The original paper proposes three versions of the score, which include different sets of arteries in the computation of the global score:

- **Central**: includes only mediastinal and lobar arteries
- **Peripheral**: includes only segmental arteries
- **Global**: includes all three levels (mediastinal, lobar, and segmental arteries)

### Obstruction-to-Degree Conversion

Each artery's obstruction value ($o_i$) is converted to a degree ($d_i$) from 0 to 5, based on thresholds proposed in the original paper:

- $d_i = 0$ if $o_i = 0\\%$
- $d_i = 1$ if $0\\% < o_i < 25\\%$
- $d_i = 2$ if $25\\% \\le o_i < 50\\%$
- $d_i = 3$ if $50\\% \\le o_i < 75\\%$
- $d_i = 4$ if $75\\% \\le o_i < 100\\%$
- $d_i = 5$ if $o_i = 100\\%$

### Score Formula

The final score is the sum of the degrees divided by the maximum possible score (i.e. all arteries obstructed at $100% \\rightarrow |A| \\times 5$ ), normalizing the result to a [0,1] scale.

$$ \\text{Mastora score} = \\sum\_{i \\in A} \\frac{d_i}{5} $$
