## Capstone-Project-2024 

### Symone Randle

### December 2nd, 2024
<br>
<br>
### Abstract \
The Open Graph Benchmark organization comprises a team of developers from many industries who collect
large datasets for machine learning algorithms. The "ogbg-molpcba" dataset contains a large collection
of graphs that represent molecular data. This project aims to determine and identify the outliers in
molecular features. The anomaly detection model used for this project is the Graph Neural Network (GNN),
utilizing the power of a Graphics Processing Unit (GPU). Given that molecules can take many
shapes, it can be difficult to determine what exactly is an anomaly.
<br>
### Methods
<br>
The "ogbg-molpcba" dataset is available on the TensorFlow catalog website, which describes where the 
data is from and where to find further information. Exploratory data analysis was conducted with the
aid of online resources, such as the OGB website. https://ogb.stanford.edu/docs/graphprop/. At first,
the AnomalyDetection model was considered for this project, but this model was found inefficient
for modeling with graph data. The GNN model runs well with the dataloader type in particular. 
Unfortunately, the same cannot be said for the advanced precision score.
<br>
### Conclusion
<br>
In conclusion, this project can initialize further studies in anomaly detection in molecular data 
by scaling for more complex machine learning methods. The model captures intricate patterns that 
represent the atoms and bonds within molecules. The GNN model used scored a 0.0102 precision score,
which is not ideal. This is potentially due to impractical anomaly criteria. Moving forward, the 
focus will be on anomalies that discover ways to progress molecular research.
<br>
### Statement of Business Value
<br>
Anomaly detection of molecules could potentially be used for drug development, ecological studies, 
research in nutrition, farming advancements, and molecular biology research. Researchers could use
this model to find which molecules stand out and why, potentially leading to new discoveries. 
Biotechnology companies should invest in data science resources because they will have the ability to
leverage data in ways that could potentially cut costs, compute complex problems efficiently, and 
position themselves as industry leaders. The advantages of data science as a tool are not fully 
realized.
<br>
### Citations
<br>
'''
# @inproceedings{DBLP:conf/nips/HuFZDRLCL20,
#   author    = {Weihua Hu and
#                Matthias Fey and
#                Marinka Zitnik and
#                Yuxiao Dong and
#                Hongyu Ren and
#                Bowen Liu and
#                Michele Catasta and
#                Jure Leskovec},
#   editor    = {Hugo Larochelle and
#                Marc Aurelio Ranzato and
#                Raia Hadsell and
#                Maria{-}Florina Balcan and
#                Hsuan{-}Tien Lin},
#   title     = {Open Graph Benchmark: Datasets for Machine Learning on Graphs},
#   booktitle = {Advances in Neural Information Processing Systems 33: Annual Conference
#                on Neural Information Processing Systems 2020, NeurIPS 2020, December
#                6-12, 2020, virtual},
#   year      = {2020},
#   url       = {https://proceedings.neurips.cc/paper/2020/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html},
#   timestamp = {Tue, 19 Jan 2021 15:57:06 +0100},
#   biburl    = {https://dblp.org/rec/conf/nips/HuFZDRLCL20.bib},
#   bibsource = {dblp computer science bibliography, https://dblp.org}
# }


# @misc{TFDS,
#   title = {{TensorFlow Datasets}, A collection of ready-to-use datasets},
#   howpublished = {\url{https://www.tensorflow.org/datasets}},
# }

# https://pytorch.org/docs/stable/index.html
# https://medium.com/@techtes.com/graph-neural-networks-gnns-for-anomaly-detection-with-python-5dfc67e35acc
# https://ogb.stanford.edu/docs/graphprop/
'''
