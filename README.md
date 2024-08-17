# Automated-Detection-and-Classification-of-Exoplanets-Using-Deep-Learning-on-Light-Curve-Data


***Disclaimer:***

This project on Automated Detection and Classification of Exoplanets Using Deep Learning on Light Curve Data is a research and educational endeavor. The results and insights derived from this project are based on the specific dataset used, model architecture, and training process, and may not be universally applicable to all exoplanet detection scenarios. Users and researchers are encouraged to consider the following limitations and to validate the findings independently before applying them in critical or real-world scenarios.

***Limitations:***

**Dataset-Specific Results:**<br>
The model's performance is closely tied to the dataset it was trained on, which primarily consists of light curve data from specific space missions such as Kepler or TESS. The accuracy and generalization of the model may vary when applied to data from other sources or with different characteristics.

**Model Sensitivity to Noise:**<br>
The deep learning model may be sensitive to noise in the light curve data, such as instrument errors or cosmic interference. This could lead to false positives or negatives in exoplanet detection, particularly in cases where the light curves are noisy or incomplete.

**Simplification of Complex Phenomena:**<br>
The model assumes that light curves can be accurately interpreted by a deep learning algorithm without accounting for all astrophysical phenomena. This simplification may lead to misclassification in cases where other astrophysical events (e.g., stellar flares, binary stars) mimic or obscure the signals of exoplanets.

**Limited Feature Scope:**<br>
The model focuses on a predefined set of features extracted from light curves. Additional factors, such as stellar activity, orbital mechanics, or multi-planet systems, are not explicitly modeled and could affect the detection and classification outcomes.

**Computational Constraints**:<br>
Deep learning models are computationally intensive, and the quality of results may depend on the computational resources available. Limited computational power could lead to suboptimal training, affecting the model's performance.

**Generalization to New Data:**<br>
While the model may perform well on the dataset it was trained on, its ability to generalize to new, unseen data (especially from different stellar environments or instruments) is not guaranteed. Users should exercise caution when applying the model to data outside the scope of the training set.

**Ethical and Interpretive Considerations:**<br>
The interpretation of results produced by deep learning models should be conducted with caution. The "black-box" nature of these models means that their decision-making process is not always transparent, which can pose challenges for scientific interpretation and ethical use.

**Dependence on Preprocessing Steps:**<br>
The model's effectiveness is dependent on the quality and consistency of the preprocessing steps applied to the light curve data. Any deviation in these steps could impact the model's ability to accurately detect and classify exoplanets.


**Dependencies:**
 - pandas
 - numoy
 - scikit-learn
 - tensorflow
 - matplotlib
