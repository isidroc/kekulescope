# KekuleScope

This repository contains the 33 data sets, code and scripts needed to generate the compound activity prediction models reported in:

[_KekuleScope: prediction of cancer cell line sensitivity and compound potency using convolutional neural networks trained on compound images_](https://doi.org/10.1186/s13321-019-0364-5)

Isidro Cortés-Ciriano and Andreas Bender

Journal of Cheminformatics, 2019, 11:41

# Dependencies
KekuleScope depends on the following python libraries:
```
scipy==1.0.0
numpy==1.15.1
joblib==0.11
ipython==7.1.1
Pillow==5.3.0
profilehooks==1.10.0
rdkit==2009.Q1-1
scikit_learn==0.20.0
torch==0.4.1.post2
torchvision==0.2.1
```

To install these libraries run:

```
pip install -r requirements.txt
```

# Running the code
To obtain help on how to run the models run:
```
python Kekulescope.py --help
```
It is assumed that the user has access to at least 1 GPU card.

# Contact
Please contact Isidro Cortés Ciriano, PhD (isidrolauscher at gmail.com) for further information or suggestions.

# Funding
This Project has received funding from the European Union’s Framework Programme For Research and Innovation Horizon 2020 (2014–2020) under the Marie Curie Sklodowska-Curie Grant Agreement No. 703543 (I.C.C.).
