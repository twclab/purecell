<p align="center"> <img src="https://github.com/twclab/purecell/blob/main/assets/main.png" width="450"/> </p>


# purecell

The repo contains the source code for our [paper](https://github.com/twclab/purecell/blob/main/assets/paper.pdf) "Purification of single-cell transcriptomics data with coreset selection" accepted ad ICML CompBio 2022.

### Basic usage

```
from purecell.purecell import PureCell
P = PureCell(anndata_object,batch_id='sample',label_id='type',n_neighbors=10,n_pcs=20)
indices,nodes,scores,ths = P.run(0.5)
```
