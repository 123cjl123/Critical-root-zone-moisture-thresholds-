Included Workflow
02_root_zone_vs_surface_soil_moisture_EF_decoupling.py

The organized scripts cover the main manuscript workflow. This script tests whether root zone soil moisture better explains EF anomalies than surface soil moisture under dry surface conditions. It separates cases where the surface is dry but the root zone remains relatively wet from cases where both layers are dry. It then compares EF anomalies across aridity zones and produces statistical summary plots. This analysis supports the argument that root zone moisture contains functional information that surface moisture alone can miss.

03_percentile_sensitivity_breakpoint_detection.py

This script performs percentile sensitivity testing for threshold detection. It fits a linear-plateau model between root zone soil moisture and EF anomalies under growing-season conditions. It repeats the breakpoint analysis under different lower-boundary percentile choices and summarizes model performance using R2, standard error, and threshold estimates. This script is used to justify the percentile choice for the final threshold analysis.

04_global_threshold_mapping_20th_percentile.py

This script generates the global map of critical root zone soil-moisture thresholds. It applies the selected 20th-percentile boundary method at each grid cell, fits the linear-plateau model, filters unreliable fits, and exports both raw and spatially filled threshold maps. This script produces the main spatial threshold product used in later analyses.

05_threshold_patterns_by_forest_type.py

This script summarizes critical moisture thresholds by forest functional type. It overlays the global threshold map with forest-type classes and compares threshold distributions among evergreen broadleaf forest, deciduous broadleaf forest, evergreen needleleaf forest, deciduous needleleaf forest, and mixed forest. It is used to evaluate whether threshold behavior differs among major forest types.

06_threshold_patterns_by_climate_zone.py

This script summarizes critical moisture thresholds across broad climate or biogeographic zones. It groups threshold values by climate-zone masks and compares their distributions. It supports the interpretation that threshold sensitivity varies across environmental settings.

07_independent_SIF_validation_for_root_zone_mechanism.py

This script uses solar-induced chlorophyll fluorescence (SIF) as an independent vegetation-activity metric to validate the root zone mechanism. It compares SIF responses under contrasting surface and root zone moisture states, using the same logic as the EF-based decoupling analysis. It helps test whether the root zone signal is also reflected in vegetation physiological activity.

08_XGBoost_SHAP_attribution_model_training.py

This script trains XGBoost models to explain spatial variation in critical soil-moisture thresholds. It uses climate, vegetation, and soil variables as predictors, trains separate models for different aridity zones and forest types, calculates SHAP values, and saves model packages for later plotting. This script provides the machine-learning attribution foundation.

09_SHAP_attribution_summary_plotting.py

This script loads the trained XGBoost and SHAP model packages and generates attribution summary figures. It ranks predictor importance and visualizes how climate, vegetation, and soil factors contribute to spatial variation in critical thresholds. It is the plotting step following model training.

10_Pearson_correlation_matrix_figure.py

This script generates Pearson correlation matrix figures for threshold values and explanatory variables. It provides a simpler statistical diagnostic alongside the machine-learning attribution results. It is used to show pairwise relationships among the threshold and environmental drivers.

11_SHAP_feature_interaction_matrix.py

This script produces SHAP interaction matrix figures. It evaluates how pairs of predictors jointly influence the threshold model, helping identify interactions among climate, vegetation, and soil controls. It complements the main SHAP importance analysis by focusing on combined driver effects.

12_CMIP6_future_hydraulic_failure_risk_mapping.py

This script maps future hydraulic failure risk under CMIP6 scenarios. It compares projected soil moisture with the historical critical threshold map, calculates the frequency or intensity of hydraulic failure risk, applies forest and temperature constraints, and produces future risk maps. An optional IDW interpolation step can be used to fill spatial gaps.

13_CMIP6_future_hydraulic_failure_risk_trend_line_plotting.py

This script generates trend-line figures for future hydraulic failure risk. It summarizes projected risk trajectories through time and compares future scenarios. It is used to support the manuscript section on future changes in forest hydraulic failure risk.

Excluded Preprocessing

The following scripts are not included in this folder:

Raw data download scripts.
NetCDF merging scripts.
Original latent-heat and sensible-heat flux processing scripts.
Evaporative-fraction preprocessing scripts.
Soil-moisture anomaly generation scripts.
Raster resampling and format-conversion scripts.
Intermediate table construction scripts for machine-learning input.
Temporary diagnostic and testing scripts.

These steps belong to the raw-data processing workflow rather than the core analysis workflow. They can be made available from the authors upon reasonable request.

Workflow Order

The typical analysis order is:

Calculate observed EF trends.
Compare root zone and surface soil-moisture controls on EF.
Test percentile sensitivity for threshold detection.
Generate the global critical-threshold map.
Summarize thresholds by forest type and climate zone.
Validate the root zone mechanism using SIF.
Train XGBoost attribution models and plot SHAP results.
Generate correlation and interaction diagnostics.
Map future hydraulic failure risk.
Plot future hydraulic failure risk trajectories.
Notes
Scripts 02, 04, and 07 were selected from the updated validation and threshold-analysis workflow.
The folder is intended for code review, manuscript documentation, and reproducibility support.
The folder is not a fully portable software package until the empty input and output paths are configured.
The scripts are intended for manuscript documentation, code review, and reproducibility support.
The folder is not fully executable until users provide the required input and output paths.
Preprocessing scripts and raw-data handling workflows are excluded from this organized code folder and can be provided upon reasonable request.
