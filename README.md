# Contents
1. [Introduction](#introduction)
2. [How to Run](#how-to-run)
3. [Citation](#citation)
4. [License](#license)

# Introduction

On this web page, we provide the Python implementation of the gender classification method proposed in our paper titled '[Gait-Based Gender Classification Using a Correlation-Based Feature Selection Technique](https://doi.org/10.9708/jksci.2023.28.04.041).' In this study, we proposed a method to select features that are useful for gender classification using a correlation-based feature selection technique. To demonstrate the effectiveness of the proposed feature selection technique, we compared the performance of gender classification models before and after applying the proposed feature selection technique using a three dimensional gait dataset available on the Internet. Eight machine learning algorithms applicable to binary classification problems were utilized in the experiments. The experimental results showed that the proposed feature selection technique can reduce the number of features by 22, from 82 to 60, while maintaining the gender classification performance.

# How to Run

## 1. Dataset Preparation

We utilized the Kinect Gait Biometry Dataset which consists of data from 164 individuals walking in front of an Xbox 360 Kinect Sensor. If you wish to download this dataset, please click [here](https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor) to access the download link. To obtain gender information, you will also need to download the Gender and Body Mass Index (BMI) Data for the Kinect Gait Biometry Dataset. Please click [here](https://www.researchgate.net/publication/308929259_Gender_and_Body_Mass_Index_BMI_Data_for_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor) to access the download link for the gender data.

## 2. Data Transformation (from .txt to .npz)

The skeleton sequences in the Kinect Gait Biometry Dataset are stored as text files with the filename extension .txt. To facilitate data handling, we converted the data format of each sequence from .txt to .npz. Please run 'step1_data_transformation_from_txt_to_npz.py.' After running the .py file, you will obtain the npz version of each sequence.

## 3. Feature Extraction

Please execute 'step2_feature_extraction.py.' After running the .py file, you will obtain an npz file containing feature vectors.

## 4. Correlation-Based Feature Selection

Please execute 'step3_feature_selection.py.' After running the .py file, you will obtain a figure. The figure shows the absolute value of the Pearson correlation coefficient for each pair of (Gait Feature, Gender Label).

## 5. Performance Evaluation 

Please execute the following .py files:
* 'step4_ml.py.'
* 'step5_mlp.py.'

Running the .py files mentioned above will display the evaluation results for each machine learning model.

# Citation

Please cite the following papers in your publications if they contribute to your research.

```
@article{kwon2021joint,
  author={Kwon, Beom and Lee, Sanghoon},
  journal={IEEE Access},
  title={Joint Swing Energy for Skeleton-Based Gender Classification},  
  year={2021},
  volume={9},
  pages={28334--28348},  
  doi={10.1109/ACCESS.2021.3058745}
}
```
Paper link: [Joint Swing Energy for Skeleton-Based Gender Classification](https://doi.org/10.1109/ACCESS.2021.3058745)

```
@article{kwon2024gait,
  author={Kwon, Beom},
  journal={Journal of The Korea Society of Computer and Information},
  title={Gait-Based Gender Classification Using a Correlation-Based Feature Selection Technique},
  year={2024},
  volume={29},
  number={2},
  pages={41--51},
  doi={10.9708/jksci.2023.28.04.041}
}
```
Paper link: [Gait-Based Gender Classification Using a Correlation-Based Feature Selection Technique](https://doi.org/10.9708/jksci.2023.28.04.041)

# License

Our codes are freely available for non-commercial use.
