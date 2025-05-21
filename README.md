# Epileptic Seizure Analysis: Asymmetries in Causality using Convergent Cross Mapping
_Degree Project in Computer Science and Engineering, First cycle, 15 credits_

## 📄 Abstract
Exploring brain connectivity is essential to gain a deeper understanding of epileptic seizure dynamics. Particularly, effective connectivity is used for detecting causal relationships between neural regions, with Granger Causality (GC) being a widely recognized measurement. However, identifying the specific EEG channels that are causally involved in epilepsy remains a challenge. Convergent Cross Mapping (CCM) has been developed to address these limitations of GC. Despite the potential of CCM, there have been minimal real applications in epilepsy. Additionally, previous research suggests that there is a stronger directional causal influence on certain brain regions during seizures. Therefore, this study explores causal asymmetries across channels in epilepsy by applying CCM to EEG data. The methodology of this study involves  preprocessing the data from the CHB-MIT dataset, passing it into the CCM algorithm, tuning the CCM parameters, and evaluating the resulting causality across non-seizure, preictal and ictal states through asymmetry measures. The results indicate that causality patterns are generally more asymmetric during pre-seizure and seizure activity compared to non-seizure activity. Furthermore, results from individual channels suggest that channels 20 and 21, as well as channels 6 and 12 most consistently exhibit the highest asymmetry in causality for pre-seizure and seizure activity, respectively, which is partially consistent with previous findings. It is concluded that CCM can be applied, with the improvements of scalable methods, to identify potential EEG channels that are important for underlying directional connectivity involved in seizure dynamics.

## 🗂️ Table of Contents

- [Abstract](#-abstract)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Thesis Document](#-thesis-document)

## 🏗️ Project Structure
root/
├── src/
│   ├── preprocess_data.py
│   ├── process_CCM_subjects.py
│   └── eval_CCM_subjects.py
├── docs/
│   ├── thesis.pdf
│   └── references.bib
├── README.md
└── .gitignore

## ✅ Prerequisites

<!-- List of tools, libraries, versions -->

## ⚙️ Installation

<!-- Steps to clone, set up environment, install dependencies -->

## 🚀 Usage

<!-- Commands or steps to run the project/demo -->

## 📘 Thesis Document
[📄 Read the Thesis](docs/thesis.pdf)

