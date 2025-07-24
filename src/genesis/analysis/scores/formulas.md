# Scoring Formulas

These scores are a measure of pulmonary embolism severity based on arterial obstruction.
The obstruction attribute can be chosen through the `--obstruction-attr` flag.

&#160;

## Mastora Score

The Mastora score calculation involves identifying relevant arteries, collecting their obstruction values, and then applying a specific formula based on whether percentages or degrees are used. Let $N = |A|$ be the number of selected arteries.

### Artery Selection and Obstruction Collection

- **Select Arteries**: Identify a set of arteries, $A$, based on their anatomical level (mediastinal, lobar, segmental).
- **Collect Obstructions**: For each artery $i \\in A$, get its obstruction value, $o_i$, which is a float in the range $[0, 1]$.

### Score Calculation

#### Percentage-based Calculation

If using percentages directly (`--use-percentage` flag):

$$ \\text{Mastora Score} = \\frac{\\sum\_{i \\in A} o_i}{N} $$

#### Degree-based Calculation

If not using percentages, each obstruction value $o_i$ is first converted to a degree $d_i$ on a scale from 1 to 5:

$$ d_i = \\lfloor \\frac{o_i}{0.25} \\rfloor + 1 $$

The final score is the sum of these degrees normalized by the maximum possible sum of degrees ($N \\times 5$):

$$ \\text{Mastora Score} = \\frac{\\sum\_{i \\in A} d_i}{5N} $$

&#160;

## Qanadli Score

### Traversal and Artery Scoring

The algorithm performs a depth-first search of the arterial tree. The scoring depends on the artery type:

- **Mediastinal and Lobar Arteries**: If an artery's obstruction ($o_i$) is greater than a minimum threshold (set by `--min-obstruction-thresh`), it is scored and its traversal path **stops**. The score is weighted by the total number of downstream segmental arteries ($w_i$), and these downstream arteries are not evaluated individually. If the obstruction is below the threshold, the artery is not scored, and the traversal continues to its children.
- **Segmental Arteries**: A segmental artery is only evaluated if the traversal reaches it (meaning its parent arteries were not considered obstructed). It is then scored individually with a weight ($w_i$) of 1.

### Obstruction-to-Degree Conversion

Each included artery's obstruction value ($o_i$) is converted to a degree ($d_i$) from 0 to 2, based on two thresholds (`--min-obstruction-thresh` and `--max-obstruction-thresh`, with default values of 0.25 and 0.75).

- $d_i = 0$ if $o_i < T\_{min}$
- $d_i = 1$ if $T\_{min} \\le o_i < T\_{max}$
- $d_i = 2$ if $o_i \\ge T\_{max}$

### Score Calculation

The final score is the sum of weighted degrees divided by the maximum possible sum, normalizing the result to a scale of 0 to 1.

$$ \\text{Qanadli Score} = \\frac{\\sum\_{i \\in A} w_i \\cdot d_i}{2 \\sum\_{i \\in A} w_i} $$
