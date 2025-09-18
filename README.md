# PCa_ML_Project
This is the repository of ML based Biomarker Identification from Prostate Cancer (PCa) Microarray Gene Expression Data.

Paper Title: Leveraging Machine Learning for Severity Level-wise Biomarker Identification in Prostate Cancer Microarray Gene Expression Data
Submitted to Biomedicines MDPI journal. 

Authors: Ahmed Al Marouf, Tarek A. Bismar, Sunita Ghosh, Jon G. Rokne, Reda Alhajj
Affiliation: University of Calgary, Alberta, Canada

Tentative Abstract: Prostate cancer is the most commonly occurring cancer amongst men. The detection and treatment of this cancer is therefore of great importance. The severity level of this cancer, which is established as a score in the Gleason Grading Group (GGC), guides the treatment of the cancer. In this paper traditional machine learning (ML) classification methods such as Decision Tree (DT), Random Forest (RF), Support Vector Machine (SVM), and XGBoost (XGB), which have recently been shown to accurately identifying biomarkers for computational biology, are leveraged for finding potential biomarkers for the different GGC scores. A ML framework that maps Gleason Grading Group (GGG) into five severity levels: low, intermediate-low, intermediate, intermediate-high and high has been developed using the above methods. The microarray data for this ML method has been derived from immunohistochemical tests. The study includes severity level-wise biomarker identification incorporating missing value imputation, class imbalance handling using the SMOTE-Tomek link method, and stratified k-fold validation to ensure robust biomarker selection. The framework is evaluated on prostate cancer tissue microarray gene expression data from 1119 samples. A combination of high aggressive and low aggressive signatures are used in four experimental setups. The results demonstrate the effectiveness of the approach in distinguishing critical biomarkers with highly accurate models obtaining 96.85\% accuracy using XGBoost method.

Proposed Methodology (Figure-1) ![alt text](https://github.com/AhmedAlMarouf/PCa_ML_Project/blob/Plots/Results/fig1_proposed_method.png)
Correlation Plot (Figure-2) ![alt text] (https://github.com/AhmedAlMarouf/PCa_ML_Project/blob/Plots/Results/fig2_correlation.png)
Accuracy Comparison  ![alt text](https://github.com/AhmedAlMarouf/PCa_ML_Project/blob/Plots/Results/Picture1.png)
Precision Comprison  ![alt text](https://github.com/AhmedAlMarouf/PCa_ML_Project/blob/Plots/Results/Picture2.png)
Recall Comparison  ![alt text](https://github.com/AhmedAlMarouf/PCa_ML_Project/blob/Plots/Results/Picture3.png)
F1-Score Comparison   ![alt text](https://github.com/AhmedAlMarouf/PCa_ML_Project/blob/Plots/Results/Picture4.png)
