# CMPE255Project
# Citi Bike Station Availability Forecasting

A machine learning project that forecasts **next-hour Citi Bike station utilization** using historical trip activity, station capacity, geographic information, calendar patterns, and New York City weather data.

The goal is to anticipate when a station may become unusually full or empty, supporting better bike rebalancing decisions and improving the likelihood that riders can find an available bike or dock.

## Project Overview

Bike-share demand changes throughout the day and varies by station, neighborhood, weather, and day of the week. This project converts Citi Bike trip records into a station-hour time series and trains regression models to predict each station's utilization ratio one hour into the future.

The final pipeline includes:

* Trip-data cleaning and station-name normalization
* Station-hour aggregation of arrivals and departures
* Station-capacity integration using GBFS metadata
* A capacity-aware station utilization proxy
* Weather and calendar feature engineering
* Leakage-safe lag and rolling features
* Geographic station clustering with K-Means
* Weather dimensionality reduction with PCA
* Chronological train, validation, and test splits
* Baseline and machine-learning model comparison
* Prediction clipping to physically valid utilization ranges

## Problem Definition

### Target

The primary prediction target is:

```text
utilization_ratio_next_hour
```

This represents the estimated proportion of a station's bike capacity that will be occupied during the following hour.

Predictions are bounded to the valid interval `[0, 1]`. Availability-style count predictions are similarly constrained to `[0, station_capacity]`.

### Why It Matters

A station with utilization near:

* **0** may have few or no bikes available.
* **1** may have few or no open docks available.

Accurate short-term forecasts could help operators identify stations that require rebalancing before service quality deteriorates.

## Data Sources

The modeling workflow combines three categories of data:

1. **Citi Bike trip records**

   * Ride start and end timestamps
   * Start and end station identifiers
   * Station coordinates
   * Member/casual rider information where available

2. **Station metadata**

   * Station identifiers and names
   * Latitude and longitude
   * Station capacity from Citi Bike GBFS data
   * Flags identifying raw and imputed capacity values

3. **NYC weather data**

   * Hourly weather measurements
   * Aggregated weather features used during modeling

Large raw Citi Bike trip files are not included in this repository. See the setup section for the expected workflow.

## Modeling Workflow

### 1. Data Cleaning

The cleaning pipeline standardizes schemas across trip files, parses timestamps, removes invalid records, normalizes station information, and creates a combined analysis-ready dataset.

The main cleaning script is:

```text
build_cleaned_combined_v2.py
```

Its generated outputs include:

```text
cleaned_combined.parquet
station_lookup.csv
cleaning_summary.json
```

### 2. Station-Hour Aggregation

Individual rides are aggregated into hourly activity for each station. The pipeline derives features such as:

* Hourly departures
* Hourly arrivals
* Net station flow
* Cumulative net flow
* Estimated bikes
* Available docks
* Utilization ratio

### 3. Feature Engineering

The project creates features across several categories.

#### Temporal features

* Hour of day
* Day of week
* Weekend indicators
* Cyclical sine/cosine encodings

#### Historical station features

* Lagged station utilization
* Lagged estimated bike counts
* Lagged available dock counts
* Rolling averages and recent-flow statistics

All predictive historical features are shifted so that future information is not used to predict the past.

#### Geographic features

Stations are grouped with K-Means using geographic coordinates. These clusters provide a compact representation of station location and neighborhood-level demand patterns.

#### Weather features

Weather variables are merged at the hourly level. Principal component analysis is used to reduce correlated weather measurements while retaining most of their variance.

#### Capacity features

Station capacity is used to create bounded, interpretable availability estimates. The pipeline also tracks whether capacity values came directly from metadata or were imputed.

### 4. Leakage-Safe Evaluation

The data is split chronologically rather than randomly:

```text
Training period -> Validation period -> Test period
```

Models are selected using validation results. The selected approach is then retrained on the combined training and validation data and evaluated once on the held-out test set.

This setup more closely reflects real forecasting, where a model predicts future observations from past data.

## Models

The primary notebook compares:

* Naive persistence baseline
* Linear Regression
* Random Forest
* Deeper Random Forest configuration

## Results

The strongest held-out test performance came from the deeper Random Forest model.

| Model                  |       RMSE |        MAE |         R² |
| ---------------------- | ---------: | ---------: | ---------: |
| Random Forest (deeper) | **0.0963** | **0.0660** | **0.8469** |
| Random Forest          |     0.1011 |     0.0690 |     0.8312 |
| Linear Regression      |     0.1237 |     0.0885 |     0.7475 |
| Naive baseline         |     0.1450 |     0.0939 |     0.6530 |

These metrics are measured on the held-out test split for `utilization_ratio_next_hour`.

The deeper Random Forest reduced RMSE by approximately **33.6%** relative to the naive baseline and explained approximately **84.7%** of the variance in next-hour station utilization.

## Repository Structure

```text
CMPE255Project/
├── CitiBike_2025_EDA_Modeling_v7_capacity_fix (2).ipynb
├── CitiBike_Availability_Improvement_State_Model_v2 (1).ipynb
├── CitiBike_Availability_Improvement_v7 (1).ipynb
├── CitiBike_Follow_On_Experiments_GPU (1).ipynb
├── CitiBike_Follow_On_Experiments_v7 (1).ipynb
├── CitiBike_Next_Steps_Journey_capacity_merge_fixed (1).ipynb
├── CitiBike_Next_Steps_Journey_v7.ipynb
├── Data_Cleaning.ipynb
├── build_cleaned_combined_v2.py
├── cleaned_combined.parquet
├── cleaning_summary.json
├── station_lookup.csv
├── nyc2025weatherdata.csv
├── nyc2025weatherdataAggregate.csv
└── weather_exploring.ipynb
```

### Recommended Starting Point

The primary end-to-end modeling notebook is:

```text
CitiBike_2025_EDA_Modeling_v7_capacity_fix (2).ipynb
```

It contains the capacity-aware target construction, feature engineering, exploratory analysis, validation workflow, and final test evaluation.

The remaining notebooks document exploratory analysis and follow-on experiments.

## Setup

### Requirements

* Python 3.10+
* Jupyter Notebook or JupyterLab

Install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn pyarrow requests jupyter
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn matplotlib seaborn pyarrow requests jupyter
```

Depending on the notebook being run, additional packages may be required.

## Running the Project

1. Clone the repository.

```bash
git clone https://github.com/LarryTNguyen/CMPE255Project.git
cd CMPE255Project
```

2. Create and activate a virtual environment.

3. Install the dependencies listed above.

4. Add the required raw Citi Bike trip files to the location expected by the cleaning script or update its input path.

5. Run the cleaning pipeline if regenerating the combined dataset.

```bash
python build_cleaned_combined_v2.py
```

6. Start Jupyter.

```bash
jupyter notebook
```

7. Open the primary modeling notebook:

```text
CitiBike_2025_EDA_Modeling_v7_capacity_fix (2).ipynb
```

8. Review the notebook's configuration and data paths before running all cells.

## Key Technical Decisions

### Capacity-Aware Target

Trip counts alone do not directly represent bike or dock availability. This project combines station capacity with cumulative net flow to construct an interpretable availability proxy.

### Chronological Splitting

Random train/test splits can leak future demand patterns into training. A chronological split provides a more realistic estimate of forecasting performance.

### Lagged and Rolling Features

Current and recent station conditions are highly informative, but they must be shifted before modeling. The pipeline constructs these features so that each prediction uses only information available at prediction time.

### Bounded Predictions

A utilization ratio below 0 or above 1 is not physically meaningful. Predictions are clipped to valid ranges before interpretation.

## Limitations

* Estimated availability is reconstructed from trip flow and capacity rather than continuous historical station-status snapshots.
* Capacity metadata may be missing or require imputation for some stations.
* Special events, service outages, station removals, and temporary closures are not fully modeled.
* Performance may vary across stations and time periods even when aggregate metrics are strong.
* The current evaluation focuses on 2025 data and should be validated on later periods before operational use.

## Future Improvements

* Use historical GBFS station-status snapshots as direct availability labels
* Add holiday, event, transit, and neighborhood features
* Evaluate gradient boosting models such as XGBoost, LightGBM, or CatBoost
* Compare global models with station-specific or cluster-specific models
* Add prediction intervals and uncertainty estimates
* Evaluate errors by station, hour, weekday, and demand level
* Develop a dashboard or API for real-time station-risk forecasts
* Automate data ingestion, model retraining, and monitoring

## Technologies

* Python
* Pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* Parquet / PyArrow
* Citi Bike GBFS metadata

## Contributors

This project was completed for **CMPE 255: Data Mining**.

Add team member names and individual contributions here when applicable.

## Acknowledgments

* Citi Bike for publicly available trip and GBFS station data
* Public NYC weather-data sources used in the analysis
* San Jose State University CMPE 255 course instruction and project guidance

## License

No license has currently been specified. Add a license before allowing reuse or redistribution of the project code.
