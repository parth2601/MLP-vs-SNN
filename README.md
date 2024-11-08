# MLP-vs-SNN
Comparative Analysis of MLP and SNN on MNIST Dataset

This project compares the performance of a Multi-Layer Perceptron (MLP) and a Spiking Neural Network (SNN) on the MNIST dataset.

Requirements

- Python 3.6 or higher
- Libraries:
  - torch
  - torchvision
  - snntorch
  - matplotlib
  - seaborn
  - scikit-learn
  - numpy

Installation

1. Clone the Repository

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. Install Dependencies

   ```bash
   pip install torch torchvision snntorch matplotlib seaborn scikit-learn numpy
   ```

   *Note:* Ensure that your PyTorch installation is compatible with your system and supports CUDA if you plan to use a GPU.

Usage

1. Open the Jupyter Notebook

   ```bash
   jupyter notebook MLP_vs_SNN.ipynb
   ```

2. Run the Notebook

   - Execute all cells sequentially.
   - The notebook will:
     - Load and preprocess the MNIST dataset.
     - Define and train both MLP and SNN models.
     - Evaluate the models and display performance metrics.
     - Generate and save plots in the project directory.


Notes

- Training Time: The SNN model may take significantly longer to train than the MLP model.
- GPU Usage: Using a GPU is recommended for training the SNN model.
- Adjustable Parameters: You can modify hyperparameters like learning rate, batch size, number of epochs, and neuron parameters in the notebook.

---

*Author: `Parth Patne`*
