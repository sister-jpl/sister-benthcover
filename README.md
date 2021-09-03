# sister-benthcover

Classify benthic reflectance into four benthic cover classes:

- Algae
- Coral
- Mud/sand
- Seagrass

## Installation

```bash
pip -r requirements.txt
```

## Use

```bash
python benthcover.py benthic_reflectance_image output_directory
```

Optional arguments:

- `--verbose`: default = False
- `--model`: Classifer model, default = ranfor, options:
                                    - ranfor (random forest)
                                    - logreg (logistic regression)
- `--prob`: Export class probablities, default = False

