# Predictive Maintenance Project

This repository contains code and resources for a predictive maintenance solution.

## Project Structure

- `predictive_maintenance/` - Root folder for the application
  - `data/` - Contains dataset(s)
    - `engine_data.csv` - Sample engine sensor data
  - `deployment/` - Deployment related scripts
    - `app.py` - Flask application for serving predictions
    - `Dockerfile` - Container configuration
    - `requirements.txt` - Python dependencies for deployment
  - `hosting/` - Hosting utilities
    - `hosting.py` - Code to integrate with hosting environment
  - `model_building/` - Model development scripts
    - `data_register.py` - Data registration utilities
    - `prep.py` - Data preparation routines
    - `train.py` - Model training script

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip` package manager

### Installation

```bash
cd predictive_maintenance
pip install -r requirements.txt
```

### Development

- Use the scripts in `model_building` to prepare data and train models.
- Launch the Flask app via `deployment/app.py` for serving predictions.

## Usage

1. Prepare data and train a model:
   ```bash
   python model_building/prep.py
   python model_building/train.py
   ```
2. Start the API service:
   ```bash
   python deployment/app.py
   ```
3. Make requests to the API as documented inside `deployment/app.py`.

## Contributing

Feel free to open issues or submit pull requests.

## License

MIT License