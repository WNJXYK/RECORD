# RECORD

This is a demo for the paper: "RECORD: Resource Constrained Semi-Supervised Learning under Distribution Shift".
 
##### Prerequisites

* Python 3.7
* Pytorch
* scikit-learn
* h5py
* matplotlib

##### Folders

* src: Python source code.
* data: Placeholder for the dataset. Please download the dataset from [Google Drive](https://drive.google.com/drive/folders/1rWSbgKLkBMI1ZeBDNaXBvnkDIwBP5X8U).
* logs: Placeholder for the running logs.
* images: Placeholder for the line charts.

##### To Run

For example, you can run the following command in the root path:
```
python ./src/run.py
```
The result will be saved in logs folder with a line chart saved in the images folder.

In this demo, we prepared four benchmark data sets for the distribution shift and three implements of semi-supervised learning methods. 
You can also use command `python ./src/run.py -h` to list the usages.

Contact wnjxyk@gmail.com for more questions.