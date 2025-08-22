# Scoring Formulas

These scores measure arterial obstruction caused by pulmonary embolism, with different levels of precision.
Arterial obstruction, and the proposed scores, are hypothesized to be an indicator of pulmonary embolism severity and/or patient outcome.
The obstruction attribute can be chosen through the `--obstruction-attr` flag.

Both scores involve identifying the local obstruction for all arteries of interest, and then applying a formula to compute a global measure of obstruction based on the obstruction values of these arteries.

&#160;

## Mastora Score

The Mastora score is the most precise measure of arterial obstruction. We propose two methods for calculating the global score:
one based on the exact obstruction values (in percentages) and another based on a discrete scale of obstruction (from 1 to 5).

### Artery Selection and Obstruction Measures

- **Select Arteries**: Identify arteries of interest, $A$, based on their anatomical level (mediastinal, lobar, segmental).
- **Measure Obstructions**: For each artery $i \\in A$, get its obstruction value, $o_i$, which is a float in the range $[0, 1]$.

### Score Calculation

#### Percentage-based Calculation

If using percentages directly (`--use-percentage` flag):

$$ \\text{Mastora score} = \\frac{\\sum\_{i \\in A} o_i}{|A|} $$

#### Degree-based Calculation

If not using percentages, each obstruction value $o_i$ is first converted to a degree $d_i$ on a scale from 1 to 5:

$$ d_i = \\lfloor \\frac{o_i}{0.25} \\rfloor + 1 $$

The final score is the sum of these degrees normalized by the maximum possible score ($|A| \\times 5$):

$$ \\text{Mastora score} = \\frac{\\sum\_{i \\in A} d_i}{5\\ |A|} $$

&#160;

## Qanadli Score

### Vascular Tree Traversal and Artery Scoring

The algorithm performs a depth-first search of the arterial tree. The scoring depends on the artery type:

- **Mediastinal and Lobar Arteries**: If an artery's obstruction ($o_i$) is greater than a minimum threshold (set by `--min-obstruction-thresh`), it is considered obstructed and the traversal of its subtree stops. The score is weighted by the number of descendant segmental arteries ($w_i$), and these descendant are not traversed. If an artery's obstruction is below the threshold, the traversal continues to its children.
- **Segmental Arteries**: A segmental artery is only evaluated if the traversal reaches it (meaning its parent arteries were not considered obstructed). It is then scored individually with a weight ($w_i$) of 1.

### Obstruction-to-Degree Conversion

Each included artery's obstruction value ($o_i$) is converted to a degree ($d_i$) from 0 to 2, based on two thresholds ($T\_{min}$ / `--min-obstruction-thresh` and $T\_{max}$ / `--max-obstruction-thresh`, with default values of 0.25 and 0.75).

- $d_i = 0$ if $o_i < T\_{min}$
- $d_i = 1$ if $T\_{min} \\le o_i < T\_{max}$
- $d_i = 2$ if $o_i \\ge T\_{max}$

### Score Calculation

The final score is the sum of weighted degrees divided by the maximum possible score (i.e. all arteries obstructed above $T\_{max}$), normalizing the result to a [0,1] scale.

$$ \\text{Qanadli score} = \\frac{\\sum\_{i \\in A} w_i \\cdot d_i}{2 \\sum\_{i \\in A} w_i} $$
