# Exoplanet Detection and Habitability Analysis

## Overview

This repository houses two cutting-edge astrophysical models designed to advance our understanding of exoplanetary systems:

1. **Exoplanet Detection via Transit Photometry**
2. **Exoplanet Habitability Classification**

These models leverage state-of-the-art machine learning techniques to analyze astronomical data, contributing to the rapidly evolving field of exoplanet science.

## Exoplanet Detection Model

### Methodology

Our exoplanet detection model employs Long Short-Term Memory (LSTM) networks to analyze stellar photometric time series data, specifically light curves obtained from space-based observatories such as Kepler and TESS. This approach allows for the identification of periodic dimming events indicative of planetary transits.

### Dataset

The model is trained on a comprehensive dataset comprising 56,461 light curve samples:
- 27,994 positive samples (confirmed exoplanets)
- 28,467 negative samples (stellar variability, instrumental noise, and other astrophysical phenomena)

### Performance Metrics

- Training Accuracy: 99.24%
- Validation Accuracy: 99.65%
- Test Accuracy: 99.68%
- Precision: 99.98%
- Recall: 99.37%

### Learning Curves

![Model Accuracy and Loss](Exoplanet_latest.png)


These metrics demonstrate the model's robust capability to discriminate between genuine planetary transits and false positives, even in the presence of complex stellar variability and instrumental artifacts.

## Exoplanet Habitability Classification Model

### Methodology

Building upon our transit detection capabilities, we've developed a hybrid Convolutional Neural Network (CNN) and LSTM architecture to assess exoplanet habitability. This model analyzes a multitude of planetary and stellar parameters to classify potential habitable worlds.

### Key Parameters

- Planetary equilibrium temperature (T_eq)
- Planetary radius (R_p)
- Orbital period (P)
- Semi-major axis (a)
- Stellar effective temperature (T_eff)
- Stellar mass (M_*)
- Stellar metallicity ([Fe/H])

### Performance Metrics

- Training Accuracy: 81.0%
- Validation Accuracy: 80.0%
- Test Accuracy: 81.50%
- Total Samples: 5000

## Visualizations

### Transit Light Curve Analysis
![Light Curve Around Exoplanet](Exoplanet_graph_balanced.png)

This figure illustrates a typical exoplanetary transit light curve, showcasing the characteristic dip in stellar flux as the planet occults its host star.

### Planetary System Characteristics
![Star Mass vs Planet Mass](starmassvsplanetmass.png)

This scatter plot reveals the relationship between stellar and planetary masses, providing insights into planetary formation processes and system architectures.

### Exoplanet Density Distribution
![Planet Density Distribution](planetdensity.png)

The density distribution of exoplanets offers crucial information about planetary composition and internal structure, key factors in assessing habitability.

### Orbital Dynamics and Stellar Properties
![Orbital Period vs Star Temperature](operiodvsstarttemp.png)

This visualization explores the correlation between orbital periods and host star temperatures, helping to identify potential habitable zones within diverse stellar environments.

## Key Findings

- The absence of habitable planets within certain mass ranges highlights potential constraints on planetary habitability or limitations in current detection methods.
- Habitable exoplanets exhibit a distinct density distribution, peaking around -5.278 g/cm³ and rapidly declining at higher densities, in contrast to the broader distribution of non-habitable planets.
- The narrow range of stellar temperatures (-2.50 to 2.50 K) and orbital periods (-0.006232 to 0.0 days) for habitable exoplanets suggests highly specific conditions for potential habitability.

## Implications and Future Work

These models contribute to the broader field of exoplanetary science by:
1. Enhancing our ability to detect and characterize exoplanets in diverse stellar environments.
2. Providing a quantitative framework for assessing planetary habitability.
3. Offering insights into planetary formation, system architecture, and the potential for life beyond Earth.

Future work will focus on refining these models with data from upcoming missions, incorporating additional parameters such as atmospheric composition, and exploring the potential for detecting biosignatures.

## Dependencies

- pandas
- numpy
- scikit-learn
- tensorflow
- keras
- matplotlib
- concurrent.futures

## Data Source

[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS)

## License

This project is licensed under the [MIT License](LICENSE.md).

## Acknowledgments

I gratefully acknowledge the use of data from the Kepler and TESS missions, as well as the invaluable resources provided by the NASA Exoplanet Archive.
