# Hospital Efficiency Analysis using Data Envelopment Analysis (DEA)

This project performs a comprehensive efficiency analysis of hospitals using Data Envelopment Analysis (DEA). It combines HCAHPS (Hospital Consumer Assessment of Healthcare Providers and Systems) survey data with hospital operational data to evaluate hospital performance across multiple dimensions.

## Theoretical Foundation

Data Envelopment Analysis (DEA) is a non-parametric method used to measure the relative efficiency of a set of homogeneous Decision Making Units (DMUs) - in this case, hospitals - that convert multiple inputs into multiple outputs. The efficiency is calculated as the ratio of a weighted sum of outputs to a weighted sum of inputs.

The DEA approach distinguishes between:

- **Technical Efficiency (TE)**: How effectively a hospital converts its inputs (resources) into outputs (results)
- **Variable Returns to Scale (VRS)**: Considers scale effects (BCC model)
- **Constant Returns to Scale (CRS)**: Assumes proportional relationship between inputs and outputs (CCR model)
- **Scale Efficiency**: Indicates whether a hospital operates at its optimal size

DEA establishes an efficiency frontier consisting of the best-performing hospitals, and then measures how far each hospital is from this frontier to determine its efficiency score.

## Project Structure

```
├── data/                          # Data files
│   └── prepared_data.csv          # Preprocessed data ready for analysis
├── src/                           # Source code
│   ├── data_preparation.py        # Data loading and preprocessing
│   ├── dea_analysis.py           # Core DEA implementation
│   ├── visualization.py          # Enhanced visualizations
│   ├── advanced_analysis.py      # Advanced analysis functions
│   ├── enhanced_visualizations.py # Additional visualization functions
│   ├── run_dea_analysis.py       # Main analysis script
│   └── visualization_results/    # Generated visualizations
│       ├── efficiency_scorecard.png
│       ├── scale_efficiency.png
│       ├── advanced_radar.png
│       └── ...
├── analysis_results/             # Latest analysis outputs
│   ├── plots/                   # Generated plots
│   └── tables/                  # Analysis tables
├── archived_files/              # Archived utilities and old versions
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Features

### Core Analysis
- Data preprocessing and cleaning
- Hospital efficiency analysis using DEA methodology
- Technical efficiency calculation (VRS and CRS)
- Scale efficiency analysis
- Returns to scale classification (IRS, DRS, CRS)

### Enhanced Visualizations
1. **Efficiency Scorecard**
   - Comprehensive view of efficiency metrics
   - Scale efficiency categories distribution
   - Key performance indicators

2. **Advanced Radar Analysis**
   - Performance metrics by efficiency level
   - Multi-dimensional comparison
   - Patient care and operational metrics

3. **Scale Efficiency Analysis**
   - Distribution of efficiency categories
   - Returns to scale classification
   - Size-efficiency relationships

4. **Performance Matrix**
   - Compact relationship visualization
   - Efficiency correlations
   - Size and occupancy effects

### Context-Dependent Analysis
- Hospital size categories
- Occupancy rate analysis
- Quality metrics evaluation
- Patient satisfaction correlation

## Methodology

The DEA implementation follows these steps:

1. **Data Preparation**
   - Load and clean hospital data
   - Normalize input/output variables
   - Create efficiency categories

2. **Core DEA Analysis**
   - Calculate VRS and CRS efficiency scores
   - Determine scale efficiency
   - Classify returns to scale

3. **Advanced Analysis**
   - Generate comprehensive visualizations
   - Perform statistical analysis
   - Create summary reports

## Input and Output Variables

### Inputs (Resources)
- **Physical Resources**: Licensed beds, staffed beds
- **Human Resources**: Total surveys (proxy for staff size)
- **Financial Resources**: Bed occupancy rate (proxy for financial efficiency)

### Outputs (Results)
- **Patient Care**: Patient satisfaction, overall rating
- **Quality Metrics**: Recommendation score, staff responsiveness
- **Operational Metrics**: Nurse communication, doctor communication
- **Health Outcomes**: Cleanliness, quietness

## Requirements

- Python 3.9+
- Required packages (see requirements.txt):
  - pandas
  - numpy
  - scipy
  - pulp
  - scikit-learn
  - seaborn
  - matplotlib
  - squarify (for treemap visualizations)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hospital-efficiency-analysis.git
cd hospital-efficiency-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:
```bash
python src/run_dea_analysis.py
```

This will:
1. Load the prepared data
2. Perform DEA analysis
3. Generate enhanced visualizations
4. Create detailed analysis reports

## Key Findings

1. Scale Efficiency Distribution:
   - 55.3% operate under Decreasing Returns to Scale (DRS)
   - 34.0% operate under Increasing Returns to Scale (IRS)
   - 10.6% operate at Constant Returns to Scale (CRS)

2. Performance Metrics:
   - High efficiency hospitals show better patient satisfaction
   - Medium-sized hospitals demonstrate optimal scale efficiency
   - Strong correlation between staff responsiveness and efficiency

3. Operational Insights:
   - Bed occupancy rates significantly impact efficiency
   - Communication metrics correlate with overall efficiency
   - Size-efficiency relationship shows non-linear pattern

## Acknowledgments

- HCAHPS survey data
- Hospital operational data
- DEA methodology based on Charnes, Cooper, and Rhodes (1978) 
- Theoretical foundation follows the production theory approach
