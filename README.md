# sister-benthcover

Classify benthic reflectance into four benthic cover classes:

- Algae
- Coral
- Mud/sand
- Seagrass

Takes as input a benthic reflectance image with wavelength range 430 to 670 nm, with 10 nm spectral sampling interval.

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
- `--depth`: Filepath to depth map, used for masking > 5m, default = None
- `--model`: Classifer model type, default = logreg.
- `--level`: Classifer model hierarchical level, default = 1.
- `--prob`: Export class probablities, default = False

