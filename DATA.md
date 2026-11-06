# Data

**Dataset:** Rare Event Classification in Multivariate Time Series (paper-machine sheet breaks), released by ProcessMiner Inc.

**Citation:** Ranjan, C., Reddy, M., Mustonen, M., Paynabar, K., Pourak, K. (2018). *Dataset: Rare Event Classification in Multivariate Time Series.* arXiv:1809.10717. https://arxiv.org/abs/1809.10717

**Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0). This project is non-commercial. Derived results in `src/data/` are shared under the same licence.

**Obtaining it:** request the CSV through the link in the paper and save it as `data/raw/processminer.csv`. The raw file is git-ignored and not redistributed here. The copy used for this analysis has 18,398 rows × 63 columns (`time, y, x1..x61`) and 124 rows with `y = 1`; `pipeline/run.py` prints these counts so you can confirm a matching copy.
