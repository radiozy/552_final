# Cognitive Modeling of Heteronyms Using Neural Networks

This project models human-like disambiguation of heteronyms (words spelled the same but with different meanings/pronunciations) using neural networks. It combines insights from cognitive science and NLP to simulate context-driven interpretation of ambiguous words.

## Project Structure

- `updated_code.py`: Main Python script implementing the neural network model, data processing, and evaluation logic.
- `Final_Heteronym_Dataset.csv`: The dataset containing heteronyms, contexts, and annotated labels.
- `Cognitive Modeling of Heteronyms Using Neural Networks Report.docx`: Detailed project report describing methodology, experiments, and results.

## Setup

1. **Clone the repository** (if hosted remotely) or download the source files.
2. **Create a Conda environment** using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate heteronym-model
```

3. **Run the script**:

```bash
python updated_code.py
```

## Requirements

All required packages are listed in `environment.yml`. Core dependencies include:
- Python 3.10+
- torch
- pandas
- scikit-learn
- transformers
- numpy

## Usage

The script trains and evaluates a neural model for heteronym disambiguation. Make sure `Final_Heteronym_Dataset.csv` is in the same directory as the script.

## License

This project is for academic/research purposes.
