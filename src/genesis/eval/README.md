## CLI Commands

**Input Format**: Patient IDs are auto-padded to 4 digits and resolved to `data/graphs/{id}_graph_ep_transversal_obstruction.json`.

### ▶️ `mastora`

Score details in [formulas.md](scores/formulas.md#mastora-score).

| **Description** | Compute Mastora score for pulmonary embolism risk assessment.                                                                                                                                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usage**       | `mastora INPUT_FILE [OPTIONS]`                                                                                                                                                                                                                                         |
| **Input**       | JSON graph or patient ID (e.g. `0055`)                                                                                                                                                                                                                                 |
| **Options**     | `--use-percentage, -p` : treat degrees as percentages (0–1)<br>`--mode, -m TEXT` : artery levels (‘m’, ‘l’, ‘s’), default: `mls`<br>`--obstruction-attr, -o TEXT` : edge attribute, default: `max_transversal_obstruction`<br>`--debug, -d` : show debug visualization |
| **Examples**    | `mastora 55`<br>`mastora 0055 -p -m ml`<br>`mastora 0055 -d`                                                                                                                                                                                                           |

### ▶️ `qanadli`

Score details in [formulas.md](scores/formulas.md#qanadli-score).

| **Description** | Compute Qanadli score for pulmonary embolism risk assessment.                                                                                                                                                                       |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usage**       | `qanadli INPUT_FILE [OPTIONS]`                                                                                                                                                                                                      |
| **Input**       | JSON graph or patient ID (e.g. `0055`)                                                                                                                                                                                              |
| **Options**     | `--min-obstruction-thresh, -n FLOAT` : default `0.25`<br>`--max-obstruction-thresh, -x FLOAT` : default `0.75`<br>`--obstruction-attr, -o TEXT` : default `max_transversal_obstruction`<br>`--debug, -d` : show debug visualization |
| **Examples**    | `qanadli 55`<br>`qanadli 0055 -n 0.3 -x 0.8`<br>`qanadli 0055 -d`                                                                                                                                                                   |

### ▶️ `visualize`

| **Description** | Interactive PyVis network visualization of obstruction values.        |
| --------------- | --------------------------------------------------------------------- |
| **Usage**       | `visualize INPUT_FILE [OPTIONS]`                                      |
| **Input**       | JSON graph or patient ID (e.g. `0055`)                                |
| **Options**     | `--obstruction-attr, -o TEXT` : default `max_transversal_obstruction` |
| **Examples**    | `visualize 0055`<br>`visualize 55 -o max_transversal_obstruction`     |

### ▶️ `correlate`

| **Description** | Correlate computed scores with clinical attributes and plot.                                                                                                                                                                                                                                                                                                                 |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usage**       | `correlate SCORE_NAME ATTRIBUTE_NAME [OPTIONS]`                                                                                                                                                                                                                                                                                                                              |
| **Arguments**   | `SCORE_NAME` : `mastora` or `qanadli`<br>`ATTRIBUTE_NAME` : `bnp`, `troponin`, `risk`, `spesi`                                                                                                                                                                                                                                                                               |
| **Options**     | `--clinical-data, -c TEXT` : path to clinical CSV, default `data/PERSEVERE/clinical_data.csv`<br>`--graphs-dir, -g TEXT…` : directories to search, default `data/PERSEVERE/raw`<br>`--obstruction-attr, -o TEXT` : default `max_transversal_obstruction`<br>`--all-attributes, -a` : include all obstruction attributes<br>`--show-visualization, -v` : open plot in browser |
| **Examples**    | `correlate mastora bnp -v`<br>`correlate qanadli troponin -c custom/data.csv`<br>`correlate mastora risk -g alt/graphs -o max_transversal_obstruction_propagated`                                                                                                                                                                                                            |

#### List of obstruction attributes

| **Attribute**                            | **Description**                                  |
| ---------------------------------------- | ------------------------------------------------ |
| `max_transversal_obstruction`            | Maximum transversal obstruction value on an edge |
| `max_transversal_obstruction_propagated` | Propagated: `own = max(parent, own)`             |
| `max_transversal_obstruction_cumulated`  | Cumulated: `own = 1 - (1-parent)*(1-own)`        |

&#160;

### List of `--obstruction-attr`

| **Attribute**                            | **Description**                                                    |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `max_transversal_obstruction`            | Maximum transversal obstruction value across one edge of the graph |
| `max_transversal_obstruction_propagated` | `own_mtop = max(parent_mto, own_mto)`                              |
| `max_transversal_obstruction_cumulated`  | `own_mtoc = 1 - (1 - parent_mto) * (1 - own_mto)`                  |

### Visualization Examples

<div align="center">
  <table width="100%">
    <tr>
      <td width="50%" align="center"><b><code>visualize 55 -o max_transversal_obstruction</code></b></td>
      <td width="50%" align="center"><b><code>visualize 55 -o max_transversal_obstruction_propagated</code></b></td>
    </tr>
    <tr>
      <td width="50%" align="center"><img src="assets/original_graph.png" width="450"></td>
      <td width="50%" align="center"><img src="assets/propagated_graph.png" width="450"></td>
    </tr>
  </table>
</div>

&#160;

## NetworkX Graph Compatibility

| Function             | Underlying acyclicity | Underlying connectivity | In-degree ≤ 1 | Type       | Morgane's graphs compatibility |
| -------------------- | --------------------- | ----------------------- | ------------- | ---------- | ------------------------------ |
| `is_forest(G)`       | Yes                   | Not required            | No            | Undirected | Yes                            |
| `is_tree(G)`         | Yes                   | Yes                     | No            | Undirected | Yes                            |
| `is_branching(G)`    | Yes                   | Not required            | Yes           | Directed   | Yes                            |
| `is_arborescence(G)` | Yes                   | Yes                     | Yes           | Directed   | Yes                            |

&#160;

## Scores Comparison

```bash
correlate mastora risk -a
correlate qanadli risk -a
correlate mastora bnp -a
correlate qanadli bnp -a
correlate mastora troponin -a
correlate qanadli troponin -a
```
