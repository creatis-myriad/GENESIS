## CLI Commands

### ▶️ `eval-graph` group to analyze individual patient graphs

| **Description** | Group to chain together loading a graph with downstream tasks (e.g. scoring, visualization)                                                                                                                                                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usage**       | `eval-graph [OPTIONS] INPUT_FILE COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...`                                                                                                                                                                                                                                                          |
| **Help**        | `eval-graph --help`: Show help message and exit.                                                                                                                                                                                                                                                                                      |
| **Arguments**   | `INPUT_FILE`: Path to JSON graph, or patient ID (e.g. `0055`).                                                                                                                                                                                                                                                                        |
| **Options**     | `-g, --graphs-dirs DIRECTORY`: Directory(ies) to search for graph files. <br>`-p, --pattern TEXT`: Glob pattern to search for files within `graphs-dirs` (e.g. '\*{id}\_enriched_graph.json'). <br>`-l, --legacy-networkx-format`: Use legacy attribute names to parse NetworkX-internal graph data (i.e. 'links' instead of 'edges') |
| **Commands**    | `mastora`: Compute Mastora score on the graph. <br>`qanadli`: Compute Qanadli score on the graph.<br>`visualize`: Visualize attribute values in the graph an interactive PyVis-generated HTML.                                                                                                                                        |
| **Examples**    | `eval-graph -g data/PERSEVERE/graphs -p '*{id}*.json' -l 0055`                                                                                                                                                                                                                                                                        |

&#160;

### ▶️ `correlate` command to correlate scores across multiple patients with clinical attributes

| **Description** | Correlate computed scores with clinical attributes and plot.                                                                                                                                                                                                                                                      |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usage**       | `eval-correlate SCORE_NAME ATTRIBUTE_NAME [OPTIONS]`                                                                                                                                                                                                                                                              |
| **Arguments**   | `SCORE_NAME` : `mastora` or `qanadli`<br>`ATTRIBUTE_NAME` : `bnp`, `troponin`, `risk`, `spesi`                                                                                                                                                                                                                    |
| **Options**     | `--clinical-data, -c TEXT` : path to clinical CSV, default `data/PERSEVERE/clinical_data.csv`<br>`--graphs-dirs, -g TEXT ...` : directories to search, default `data/PERSEVERE/raw`<br>`--obstruction-attr, -o TEXT` : default `transversal_obstruction_max`<br>`--show-visualization, -v` : open plot in browser |
| **Examples**    | `eval-correlate mastora bnp -v`<br>`eval-correlate qanadli troponin -c custom/data.csv`<br>`eval-correlate mastora risk -g alt/graphs -o transversal_obstruction_max`                                                                                                                                             |

&#160;

### Chainable subcommands

#### ▶ `qanadli`

Details of how the Qanadli score is computed are provided [here](scores/formulas.md#qanadli-score).

| **Description** | Command to compute the Qanadli obstruction score on the graph(s)                                                                                                                                                                                                                                                                  |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usage**       | `GROUP_COMMAND qanadli [OPTIONS]`                                                                                                                                                                                                                                                                                                 |
| **Input**       | For how to specify input, see the [help above on the `eval-graph` command](#-eval-graph).                                                                                                                                                                                                                                         |
| **Help**        | `GROUP_COMMAND qanadli --help`: Show help message and exit.                                                                                                                                                                                                                                                                       |
| **Options**     | `-o, --obstruction-attr TEXT`: Edge attribute to use as obstruction values. <br>`-po, --partial-obstruction-thresh FLOAT`: Transversal obstruction threshold to consider a segment partially obstructed. <br>`-to, --total-obstruction-thresh FLOAT`: Transversal obstruction threshold to consider a segment totally obstructed. |
| **Examples**    | Compute Qanadli score for one patient: `eval-graph -g data/PERSEVERE/graphs -p '*{id}*.json' -l 0055 qanadli`                                                                                                                                                                                                                     |

#### ▶️ `mastora`

Details of how the Qanadli score is computed are provided [here](scores/formulas.md#mastora-score).

| **Description** | Command to compute the Mastora obstruction score on the graph(s)                                                                                                                                                                    |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usage**       | `GROUP_COMMAND mastora [OPTIONS]`                                                                                                                                                                                                   |
| **Input**       | For how to specify input, see the [help above on the `eval-graph` command](#-eval-graph).                                                                                                                                           |
| **Help**        | `GROUP_COMMAND mastora --help`: Show help message and exit.                                                                                                                                                                         |
| **Arguments**   | `{central\|peripheral\|global}`: Variant of the score to compute, that only considers obstructions in central (i.e. mediastinal and lobar), peripheral (i.e. segmental), or both (i.e. global) arteries towards in the final score. |
| **Options**     | `-o, --obstruction-attr TEXT`: Edge attribute to use as obstruction values. <br>                                                                                                                                                    |
| **Examples**    | Compute Mastora (central) score for one patient: `eval-graph -g data/PERSEVERE/graphs -p '*{id}*.json' -l 0055 mastora central`                                                                                                     |

#### ▶️ `visualize`

| **Description** | Command to visualize attribute values in the graph(s) using an interactive PyVis-generated HTML                                                                                                                                                                                                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Usage**       | `GROUP_COMMAND visualize [OPTIONS]`                                                                                                                                                                                                                                                                                                                                       |
| **Input**       | For how to specify input, see the [help above on the `eval-graph` command](#-eval-graph).                                                                                                                                                                                                                                                                                 |
| **Help**        | `GROUP_COMMAND visualize --help`: Show help message and exit.                                                                                                                                                                                                                                                                                                             |
| **Options**     | `-o, --obstruction-attr TEXT`: Edge obstruction attribute to display in the visualization. Ignored in favor of obstruction attribute used by score if `--debug-score` is also specified. <br> `-d, --debug-score [qanadli\|mastora_central\|mastora_peripheral\|mastora_global]`: Score for which to display intermediate values in the visualization, to help debugging. |
| **Examples**    | Visualize intermediate results of Qanadli score for one patient: `eval-graph -g data/PERSEVERE/graphs -p '*{id}*.json' -l 0055 qanadli visualize --score-debug qanadli`                                                                                                                                                                                                   |

#### (Debug) Visualization Examples

<div align="center">
  <table width="100%">
    <tr>
      <td width="50%" align="center"><b><code>eval-graph 0055 visualize -o transversal_obstruction_max</code></b></td>
    </tr>
    <tr>
      <td width="50%" align="center"><img src="../../../assets/transversal_obstruction_max_graph.png" width="450"></td>
    </tr>
  </table>
</div>

#### Obstruction attributes choices (`--obstruction-attr`)

| **Attribute**                 | **Description**                                                                               |
| ----------------------------- | --------------------------------------------------------------------------------------------- |
| `transversal_obstruction_max` | Maximum transversal obstruction (mto) value across one edge of the graph, i.e. a blood vessel |
