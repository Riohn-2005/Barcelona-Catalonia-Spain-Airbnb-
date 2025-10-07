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
amenities – List of amenities (Wi-Fi, AC, kitchen, etc.)
price – Price per night (string with currency).
number_of_reviews – Total reviews.
review_scores_rating – Overall rating (out of 100).
instant_bookable – Whether guests can book instantly.

## Preparing Data
```py
data = df2[[ 'host_since','host_identity_verified','property_type',
            'room_type','accommodates','bedrooms','beds','amenities','price',
             'number_of_reviews','review_scores_rating','instant_bookable','neighbourhood_group_cleansed']]
#removing the '$' 
data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)
#converting the date into only year
data['host_since'] = pd.to_datetime(data['host_since'], errors='coerce').dt.year
data= data[data['host_since'].notna()] # only 3 rows were dropped
data['instant_bookable'] = data['instant_bookable'].map({'t': 1, 'f': 0})
data['host_identity_verified'] = data['host_identity_verified'].map({'t': 1, 'f': 0})
```
