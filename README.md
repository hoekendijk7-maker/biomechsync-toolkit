# BiomechSync Toolkit

**BiomechSync Toolkit** is a Python-based research prototype for synchronising, interpolating and visualising biomechanical signals from ForceDecks (VALD), EMG (Delsys) and video recordings.

The toolkit was developed in an academic laboratory context to support biomechanical data analysis in sport science research projects.

## Overview

This repository currently includes two main scripts:

1. **BiomechSync Interpolator**  
   A Python script for synchronising ForceDecks and EMG data, interpolating signals to a common time base, and exporting a unified CSV file.

2. **BiomechSync Viewer**  
   A graphical interface for visualising synchronised biomechanical signals together with video, selecting analysis phases, and exporting selected data segments.

## Main features

- Import of ForceDecks CSV files.
- Import of EMG CSV files.
- Signal synchronisation between force and EMG data.
- Interpolation to a common time base.
- Export of synchronised CSV files.
- Visualisation of force and EMG signals.
- Video-signal synchronisation.
- Selection and export of specific analysis phases.
- Interactive graphical interface built with PyQt5 and Matplotlib.

## Research context

This software was initially developed as part of a Master's Final Project workflow and later improved for use in a Bachelor's Final Project study at the university.

The aim of the toolkit is to facilitate the exploration, synchronisation and visual inspection of biomechanical data collected in laboratory-based sport science studies.

## Development status

This software is under active development.

The current version should be considered a **research prototype**. It has been designed for academic and educational use and may require adaptation depending on the structure of the input files, devices used, and experimental protocol.

## Repository structure

```text
biomechsync-toolkit/
│
├── biomechsync_interpolator.py
├── biomechsync_viewer.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Requirements

The toolkit uses Python and the following main libraries:

- pandas
- numpy
- matplotlib
- PyQt5
- opencv-python

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Synchronise and interpolate data

Run:

```bash
python biomechsync_interpolator.py
```

The script will ask the user to select the ForceDecks CSV file, the EMG CSV file, and the destination path for the exported synchronised CSV file.

### 2. Visualise synchronised data and video

Run:

```bash
python biomechsync_viewer.py
```

The graphical interface allows the user to load the synchronised CSV file, load a video file, synchronise the video with the signals, inspect the data, select phases, and export selected segments.

## Suggested citation

If this software is used in academic work, please cite it as:

> Mena Hoekendijk, X. (2026). *BiomechSync Toolkit: Python toolkit for synchronising, interpolating and visualising biomechanical signals* [Computer software]. GitHub. https://github.com/hoekendijk7-maker/biomechsync-toolkit

## Author

Developed by **Xavier Mena Hoekendijk**  
Sports Technology Laboratory Technician  
Institut Nacional d'Educació Física de Catalunya — INEFC Lleida

## License

This project is licensed under the MIT License.

## Disclaimer

This software is provided for research and educational purposes. It is not intended for clinical diagnosis, commercial use, or validated biomechanical assessment without further verification.
