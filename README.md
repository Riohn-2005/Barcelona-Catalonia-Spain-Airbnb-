# Data analysis of the taught concept in ML Course.
```py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
import seaborn as sns
```
## Reading File
```py
df1 = pd.read_csv('/content/calendar (1).csv.gz')
df2 = pd.read_csv('/content/listings (1).csv.gz')
```
## Asked Chatgpt meaning of all the labels:
🔹 Identifiers & Metadata
id, listing_url, scrape_id, last_scraped, source – Basic listing and dataset info.
name, description, neighborhood_overview, picture_url – Listing details.

🔹 Host Information
host_id, host_url, host_name, host_since, host_location, host_about – Host details.
host_response_time/rate, host_acceptance_rate, host_is_superhost – Performance metrics.
host_listings_count, host_total_listings_count – Number of listings.
host_verifications, host_identity_verified, host_has_profile_pic – Verification info.

🔹 Location
neighbourhood, neighbourhood_cleansed, neighbourhood_group_cleansed – Area data.
latitude, longitude – Coordinates.

🔹 Property & Room Info
property_type, room_type – Type of stay.
accommodates, bathrooms, bedrooms, beds – Capacity details.
amenities – Features offered.

🔹 Price & Nights
price – Nightly rate.
minimum_nights, maximum_nights (+ variants with avg/min/max) – Stay limits.

🔹 Availability
has_availability, availability_30/60/90/365, calendar_last_scraped – Booking availability.

🔹 Reviews
number_of_reviews, reviews_per_month, first_review, last_review – Review stats.
review_scores_rating, accuracy, cleanliness, checkin, communication, location, value – Ratings.

🔹 Business Metrics
license, instant_bookable – Regulatory & booking status.
calculated_host_listings_count (+_entire_homes/private/shared) – Host activity.
estimated_occupancy_l365d, estimated_revenue_l365d – Performance estimates.




### Important labels I chose:

neighbourhood – Neighborhood name (raw).

neighbourhood_cleansed – Standardized Airbnb neighborhood. *

property_type – Type (Apartment, House, etc.)

room_type – Type of room (Entire home/apt, Private room, Shared room)

accommodates – Maximum number of guests.

bedrooms – Number of bedrooms.

beds – Number of beds.

price – Price per night (string with currency).

number_of_reviews – Total reviews.

review_scores_rating – Overall rating (out of 100).

instant_bookable – Whether guests can book instantly.

## Preparing Data
```py
data = df2[[ 'host_since','host_identity_verified','property_type',
            'room_type','accommodates','bedrooms','beds','price',
             'number_of_reviews','review_scores_rating','instant_bookable','neighbourhood_group_cleansed']]
#removing the '$' 
data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)
#converting the date into only year
data['host_since'] = pd.to_datetime(data['host_since'], errors='coerce').dt.year
# Drop rows with missing 'host_since' values (only 3 rows were removed)
data= data[data['host_since'].notna()]
# Convert 'instant_bookable' from 't'/'f' to binary 
data['instant_bookable'] = data['instant_bookable'].map({'t': 1, 'f': 0})
# Convert 'host_identity_verified' from 't'/'f' to binary
data['host_identity_verified'] = data['host_identity_verified'].map({'t': 1, 'f': 0})
```
```py
data.info() , data.isna().sum() #checking empty data
```
```
<class 'pandas.core.frame.DataFrame'>
Index: 18924 entries, 0 to 18926
Data columns (total 12 columns):
 #   Column                        Non-Null Count  Dtype  
---  ------                        --------------  -----  
 0   host_since                    18924 non-null  float64
 1   host_identity_verified        18924 non-null  int64  
 2   property_type                 18924 non-null  object 
 3   room_type                     18924 non-null  object 
 4   accommodates                  18924 non-null  int64  
 5   bedrooms                      16869 non-null  float64
 6   beds                          14873 non-null  float64
 7   price                         14912 non-null  float64
 8   number_of_reviews             18924 non-null  int64  
 9   review_scores_rating          13928 non-null  float64
 10  instant_bookable              18924 non-null  int64  
 11  neighbourhood_group_cleansed  18924 non-null  object 
dtypes: float64(5), int64(4), object(3)
memory usage: 1.9+ MB
(None,
 host_since                         0
 host_identity_verified             0
 property_type                      0
 room_type                          0
 accommodates                       0
 bedrooms                        2055
 beds                            4051
 price                           4012
 number_of_reviews                  0
 review_scores_rating            4996
 instant_bookable                   0
 neighbourhood_group_cleansed       0
 dtype: int64)
```
Conclusion:

> The dataset contains 18,924 valid entries with 12 columns.

> Categorical features such as property_type, room_type, instant_bookable, and neighbourhood_group_cleansed are complete (no missing values).

> bedrooms is missing 2,055 values (≈10.9% of total data).

> beds has 4,051 missing values (≈21.4%).

> price is missing 4,012 values (≈21.2%).

> review_scores_rating has the highest missing rate, with 4,996 missing values (≈26.4%).


### Overall Summary
```py
summary = data.describe().T
summary['missing_values'] = data.isna().sum()
summary.style.background_gradient()
```
<img width="1076" height="322" alt="Screenshot 2025-10-11 at 9 36 08 PM" src="https://github.com/user-attachments/assets/7ce3fe04-78f8-47b5-a65c-f557d04e660f" />

## Coorelation Plot
```py
plt.figure(figsize=(10,6))
data=data.select_dtypes(include='number')  # keep only numeric columns
corr = data.corr()
sns.heatmap(corr, annot= True)
```
<img width="925" height="668" alt="image" src="https://github.com/user-attachments/assets/35df48cb-e7f4-4057-90f5-0b86eb1116a2" />
Conclusion: 

> accommodates, bedrooms, and beds show strong pairwise correlations (0.66 to 0.74), indicating larger listings generally offer more rooms and beds.​

> price correlates well with accommodates (0.47), bedrooms (0.46), and price_per_accommodate (0.74), suggesting higher prices are tied to larger or more service-rich properties.

## General Data analysis

### Heatmap of Airbnb prices in Barcelona
```py
import folium
from folium.plugins import HeatMap
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ Keep only rows with valid coordinates and price
barcelona_data = df2[['latitude', 'longitude', 'price']].dropna(subset=['latitude', 'longitude', 'price'])

# 2️⃣ Optionally normalize prices (recommended)
scaler = MinMaxScaler()
barcelona_data['price_scaled'] = scaler.fit_transform(barcelona_data[['price']])

# 3️⃣ Create a base map centered on Barcelona
m = folium.Map(location=[41.3851, 2.1734], zoom_start=12)

# 4️⃣ Prepare clean heatmap data
heat_data = barcelona_data[['latitude', 'longitude', 'price_scaled']].values.tolist()

# 5️⃣ Add heatmap
HeatMap(heat_data, radius=10, blur=15, max_zoom=1).add_to(m)

# 6️⃣ Save map
m.save('barcelona_heatmap.html')

# If in Jupyter or Colab:
m
```
<img width="1217" height="729" alt="Screenshot 2025-10-12 at 2 39 27 PM" src="https://github.com/user-attachments/assets/bfa4ad1c-d7d1-4bd1-8e53-b6f71ddcdd60" />


### Average Price by Neighbourhood Group
```py
plt.figure(figsize=(10,6))
data.groupby('neighbourhood_group_cleansed')['price'].mean().sort_values(ascending=False).plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title('Average Price by Neighbourhood Group', fontsize=16, weight='bold', pad=15)
plt.xlabel('Neighbourhood Group', fontsize=12)
plt.ylabel('Average Price (USD)', fontsize=12)
```
<img width="854" height="686" alt="image" src="https://github.com/user-attachments/assets/230beacd-246c-4880-90f1-cbfefc8c3a99" />

Conclusion :

> Eixample stands out as the most expensive area, with an average price notably higher than all other neighbourhoods.​

> Sant Martí and Sants-Montjuïc follow, also commanding above-average prices, implying higher demand or premium listings in these districts.​

> Neighbourhoods like Nou Barris present the lowest average prices, well below all other areas, suggesting more budget-friendly accommodation or less central locations

### Number of Listings by Accommodation Capacity
```py
plt.figure(figsize=(10,6))
(data.groupby('accommodates')['price'].count().sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black'))
plt.title('Number of Listings by Accommodation Capacity', fontsize=16, weight='bold')
plt.xlabel('Number of Guests Accommodated', fontsize=12)
plt.ylabel('Count of Listings', fontsize=12)
```
<img width="863" height="557" alt="image" src="https://github.com/user-attachments/assets/506ce746-a549-4de8-a057-7a18b43f64d9" />

Conclusion:  

> The majority of listings accommodate 2 to 4 guests, with listings for 2 guests being the most common by a significant margin (~4,000 listings).​

> A good number of listings accommodate 1 guest and 6 guests, but these drop off quickly for larger capacities.​

> Listings accommodating more than 6 guests become increasingly rare, showing fewer than a few hundred listings per category past this point.

### Average Price by Number of Accommodates
```py
plt.figure(figsize=(12,7))
data.groupby('accommodates')['price'].mean().plot(kind = 'bar',color='skyblue', edgecolor='black')
plt.title('Average Price by Number of Accommodates', fontsize=16, weight='bold')
plt.xlabel('Number of Accommodates', fontsize=14)
plt.ylabel('Average Price ($)', fontsize=14)
```
<img width="1019" height="635" alt="image" src="https://github.com/user-attachments/assets/d08fabe8-0e4a-43b0-ba2c-ed9c1daf8435" />

Conclusion :

> The average price increases steadily with capacity; listings for 1 guest are priced lowest, and price rises consistently as the number of accommodates increases.​

> There is a sharp price jump for listings that host more than 10 guests, with extremely high averages for the largest capacities (especially at 16 accommodates, which significantly outpaces all others).​

> Listings for groups of around 12–16 guests command a substantial premium, likely due to supply scarcity and the special requirements for hosting large groups.

### Average Price Trend Over Time by Host Since
```py
plt.figure(figsize=(12,6))
data.groupby('host_since')['price'].mean().plot(kind='line', label='All Listings', color='gray',linewidth=2.9)
# Filtered average (1, 2, 4 accommodates)
filtered_data = data[data['accommodates'].isin([1, 2, 4])]
filtered_data.groupby('host_since')['price'].mean().plot(kind='line', label='Accommodates 1, 2, or 4', color='royalblue', linewidth=2.5)
plt.title('Average Price Trend Over Time by Host Since', fontsize=16, weight='bold')
plt.xlabel('Host Since (Year)', fontsize=12)
plt.ylabel('Average Price (USD)', fontsize=12)
plt.legend()
```
<img width="1009" height="553" alt="image" src="https://github.com/user-attachments/assets/97ea2adc-fd44-4494-a772-d3932a3af352" />

Conclusions :
> 2011 saw an average price spike to above $500, which is much higher than any other year. Indicating a unique event or market condition.​

> Listings for 1, 2, or 4 accommodates (the blue line) did not experience such dramatic increases, suggesting the rise was focused on larger-capacity, high-value entire homes/apartments.​

> Prices normalized for later years, confirming this was not part of a long-term trend but a short-term bubble affecting new hosts of that year—most likely triggered by demand spikes for central, higher-capacity listings.

> ## ❓❓ Why a sudden rise in price in 2011 ❓❓

### Average Price by Room Type Over Time

```py
data.groupby(['host_since', 'room_type'])['price'].mean().unstack().plot(kind='line', figsize=(10,6), marker='o',linewidth=2       )
plt.title('Average Price by Room Type Over Time')
plt.ylabel('Average Price ($)')
plt.xlabel('Year')
plt.legend(title='Room Type')
```
<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/c8add8bf-d4f7-4684-b015-46cb216a2422" />

### Average Price by Neighbourhood for Hosts in the year 2011 only
```py
# Filter only hosts in 2011
anamoly = df2[df2['host_since'].dt.year == 2011]
plt.figure(figsize=(10,6))
anamoly.groupby('neighbourhood_group_cleansed')['price'].mean().sort_values(ascending=False).plot(kind='bar')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Average Price')
plt.title('Average Price by Neighbourhood for Hosts in 2011')
```
<img width="850" height="668" alt="image" src="https://github.com/user-attachments/assets/1a395d52-da61-4d66-b7e1-b758791fe8be" />

### Price vs Time for Hosts in 2011
```py
# Convert host_since to datetime
df2['host_since'] = pd.to_datetime(df2['host_since'], errors='coerce')
# Filter only rows from the year 2011
df_2011 = df2[df2['host_since'].dt.year == 2011]
# Sort date to make the plot chronological
df_2011 = df_2011.sort_values('host_since')
plt.figure(figsize=(12,6))
plt.plot(df_2011['host_since'], df_2011['price'], marker='o',linestyle='-',alpha = 0.5)
plt.title('Price vs Time for Hosts in 2011')
plt.xlabel('Date(2011)')
plt.ylabel('Price ($)')
```
<img width="1014" height="547" alt="image" src="https://github.com/user-attachments/assets/5223bdce-4ede-4062-91a4-ec8c032789f3" />

#### Based on all the above graphs:

> The sharp rise in Airbnb prices—particularly for entire homes/apartments—was driven by the EIBTM 2011 international trade exhibition held from 29 Nov to 1 Dec 2011, which attracted thousands of visitors and exhibitors.

Key Drivers:

> Major Event Impact: EIBTM caused a surge in short-term accommodation demand, especially for entire homes/apartments, which are preferred by business travelers and groups for privacy and convenience.

> Localized Demand: Central districts like Eixample, Ciutat Vella, and Sants-Montjuïc, closest to venues and business hubs—saw prices rise by 2–2.5×, while outer areas stayed stable.

> Limited Supply: The short supply of entire homes in prime zones led to sharp, short-term price hikes.

> Unique to 2011: No comparable international event occurred in other years, explaining why prices normalized afterward.

##

### Mean vs Median Price by Room Type
```py
data.groupby('room_type')['price'].mean().sort_values(ascending=False).plot(kind = 'bar',label='Mean', alpha=0.6, color='cornflowerblue', edgecolor='black')
data.groupby('room_type')['price'].median().sort_values(ascending=False).plot(kind = 'bar', label='Median', alpha=0.6, color='orange', edgecolor='black')
plt.title('Mean vs Median Price by Room Type', fontsize=16, weight='bold')
plt.xlabel('Room Type', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend(fontsize=11)
```
<img width="575" height="559" alt="image" src="https://github.com/user-attachments/assets/529b2f82-f4ce-489b-944b-5dfabed61493" />

Conclusions

> Hotel rooms show the highest mean and median prices, with the mean slightly higher than the median, indicating presence of high-priced outliers elevating the average.​

> Entire home/apartment listings also have high mean and median prices, but the gap between mean and median is more pronounced.

> Private room and shared room categories have much lower mean and median prices, and the gaps between mean and median are less pronounced, showing these are more consistent and affordable options for travelers.​

> For all room types, the mean is higher than the median, revealing **right-skewed** price distributions, likely due to a few very expensive listings in each category.

### Median Price by Host Verified
```py
data.groupby('host_identity_verified')['price'].median().plot(kind = 'bar',color='skyblue', edgecolor='black') 
plt.title('Median Price by Host Verified (1 = YES , 0 = NO)', fontsize=16, weight='bold')
plt.xlabel('Host Verified', fontsize=14)
plt.ylabel('Median Price ($)', fontsize=14)
```
<img width="632" height="457" alt="image" src="https://github.com/user-attachments/assets/8fbe2ef0-caa3-49b9-bf90-340a91717ca6" />

Conclusion :
> Host verification status appears to influence pricing, with verified hosts generally offering lower median-priced listings.

### Median Price by Instant Bookable
```py
plt.figure(figsize=(8,6))
data.groupby('instant_bookable')['price'].median().plot(kind = 'bar',color='skyblue', edgecolor='black')
plt.title('Median Price by Instant Bookable (1 = YES , 0 = NO)', fontsize=16, weight='bold')
plt.xlabel('Instant Bookable', fontsize=14)
plt.ylabel('Median Price ($)', fontsize=14)
```
<img width="719" height="549" alt="image" src="https://github.com/user-attachments/assets/bd5d2a54-68aa-4a91-9898-12667f61f64f" />
Conclusion :
> Instant booking correlates with a higher median price

## K-Means Clustering

```py
data = data.dropna() #dropping all empty rows
#  18924 to 11378 enteries
```
```py
le = LabelEncoder()
data['neighbourhood_group_cleansed_num'] = le.fit_transform(data['neighbourhood_group_cleansed'])
data['property_type_num'] = le.fit_transform(data['property_type'])
data['room_type_num'] = le.fit_transform(data['room_type'])
```
```py
X = data[['host_since', 'host_identity_verified', 'property_type_num', 'room_type_num',
       'accommodates', 'bedrooms', 'beds', 'price',
        'number_of_reviews','review_scores_rating', 'instant_bookable','neighbourhood_group_cleansed_num']]
scalar = StandardScaler() 
X_scaled = scalar.fit_transform(X) 
```
Finding best K 
```py
a=[] #list of sum of squared distances
K=range(1,11)
for i in K:
  kmeans = KMeans(n_clusters=i, random_state=42)
  kmeans.fit(X_scaled) #FUNCTION THAT TAKES ONLY X_SCALED
  a.append(kmeans.inertia_) # sum of squared distances
plt.plot(K,a, marker='*') # found 3 is ideal
```
<img width="578" height="416" alt="image" src="https://github.com/user-attachments/assets/ada6bbd9-4f5a-4aab-bd4f-86c7142cce12" />

```py
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)
```
Observing the Cluster based on the regions

```py
d = data.groupby(["neighbourhood_group_cleansed","Cluster"]).size().unstack(fill_value=0)
sns.heatmap(d, annot=True, fmt='d', cmap='Blues')
```
<img width="669" height="432" alt="image" src="https://github.com/user-attachments/assets/a037c3af-9a31-43e9-8b43-c7037b786b0e" />

```py
plt.figure(figsize=(10,6))
sns.boxplot(x='Cluster', y='price', data=data, showfliers=False)
plt.title("Price Distribution by Cluster (without outliers)")
```
<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/9400cdb2-edc3-4c76-8001-c2c7ee233c27" />

Conclusions

> Clusters 1 and 2 are most prevalent in central neighbourhoods (Eixample, Ciutat Vella), indicating these areas have the highest diversity and density of listings across cluster types.​

> Cluster 2 likely represents higher-end or luxury listings, often concentrated in central, high-demand neighbourhoods; Cluster 0 covers more budget-friendly or economy offerings, distributed across all areas but less present in central zones.​

> The central districts (Eixample, Ciutat Vella) are not only the most active but also host the pricier segments, as reflected in the prominent role of cluster 2, while outlying areas remain focused on cluster 0 (lower-priced).

```py
plt.figure(figsize=(12,7))
scatter = plt.scatter(
    data['price'],
    data['number_of_reviews'],
    c=data['Cluster'],
    cmap='rainbow',
    s=75,              # slightly larger points
    alpha=0.7,         # soft transparency 
    )

plt.title('Price vs Number of Reviews by Cluster', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Price ($)', fontsize=14)
plt.ylabel('Number of Reviews', fontsize=14)
```
<img width="1019" height="654" alt="image" src="https://github.com/user-attachments/assets/4f89e8c5-3400-40bc-8f5a-2d4481661f06" />

We can observe that purple, light green and red indicate Culsters 0, 1 and 2 respectively

## Hypothesis testing(p-test)

For α (significance level) = 0.05

Null Hypothesis (H₀) :

There is no difference in mean price between verified and unverified hosts.
H₀ : μ(verified)  = μ(unverified)

Alternative Hypothesis (H₁):

There is a difference in mean price between verified and unverified hosts.
H₁ : μ(verified)  =!  μ(unverified)
```py
data = data.dropna()

np.random.seed(42)

A = data[data['host_identity_verified'] == 1 ]
B = data[data['host_identity_verified'] == 0 ]

obs_diff = A['price'].mean() - B['price'].mean()

dconc = np.concatenate([A['price'],B['price']])
nA = len(A)
n_perm =10000
diffs= []

for _ in range(n_perm):
  np.random.shuffle(dconc)
  a = dconc[:nA]
  b = dconc[nA:]
  diff = a.mean() - b.mean()
  diffs.append(diff)

p_val = np.mean(np.abs(diffs) >= abs(obs_diff))
print("p-values is :",p_val)
```
```
p-values is : 0.2894
```
Interpretation with your result

p-value = 0.2894

Since p > α, you fail to reject H₀.
➡️ There is no statistically significant evidence that host verification status affects listing price.
