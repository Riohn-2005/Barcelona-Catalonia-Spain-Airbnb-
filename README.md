# Barcelona-Catalonia-Spain-Airbnb-
Data analysis of the taught concept in ML Course.
```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
import seaborn as sns
```
Reading File
```py
df1 = pd.read_csv('/content/calendar (1).csv.gz')
df2 = pd.read_csv('/content/listings (1).csv.gz')
```

