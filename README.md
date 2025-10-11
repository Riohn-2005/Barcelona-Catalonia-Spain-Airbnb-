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
## Asked chatgpt meaning of all the labels:
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

### Important labels i chose:
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
data= data[data['host_since'].notna()] # only 3 rows were dropped
data['instant_bookable'] = data['instant_bookable'].map({'t': 1, 'f': 0})
data['host_identity_verified'] = data['host_identity_verified'].map({'t': 1, 'f': 0})
```
```py
data.info() , data.isna().sum() #checking empty data
```
<img width="511" height="629" alt="Screenshot 2025-10-07 at 12 28 30 PM" src="https://github.com/user-attachments/assets/538b921b-c569-4f39-b1fc-a6602f6d9f54" />

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

## General Data analysis

### Average Price by Neighbourhood Group
```py
plt.figure(figsize=(10,6))
data.groupby('neighbourhood_group_cleansed')['price'].mean().sort_values(ascending=False).plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title('Average Price by Neighbourhood Group', fontsize=16, weight='bold', pad=15)
plt.xlabel('Neighbourhood Group', fontsize=12)
plt.ylabel('Average Price (USD)', fontsize=12)
```
<img width="854" height="686" alt="image" src="https://github.com/user-attachments/assets/230beacd-246c-4880-90f1-cbfefc8c3a99" />

### Number of Listings by Accommodation Capacity
```py
plt.figure(figsize=(10,6))
(data.groupby('accommodates')['price'].count().sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black'))
plt.title('Number of Listings by Accommodation Capacity', fontsize=16, weight='bold')
plt.xlabel('Number of Guests Accommodated', fontsize=12)
plt.ylabel('Count of Listings', fontsize=12)
```
<img width="863" height="557" alt="image" src="https://github.com/user-attachments/assets/506ce746-a549-4de8-a057-7a18b43f64d9" />

Conclusion:  out of 18000 booking majority people book only for2,4,1

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

### Average Price by Room Type Over Time
```py
data.groupby(['host_since', 'room_type'])['price'].mean().unstack().plot(kind='line', figsize=(10,6), marker='o',linewidth=2       )
plt.title('Average Price by Room Type Over Time')
plt.ylabel('Average Price ($)')
plt.xlabel('Year')
plt.legend(title='Room Type')
```
<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/c8add8bf-d4f7-4684-b015-46cb216a2422" />

