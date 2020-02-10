
# coding: utf-8

# # Introduction 
# Toronto is the provincial capital of Ontario and the most populous city in Canada, with a population of more than 2.7 Million. It is an international centre of business, finance, arts, and culture, and is recognized as one of the most multicultural and cosmopolitan cities in the world.
# 
# Toronto is remarkable in diversity in its food and restaurants. Pakistani, Persian, Portuguese; aboriginal and new fusion; Japanese pancakes and Korean barbecue; fresh pasta in Little Italy, shawarmas in Greektown and the best damn dumplings in Chinatown. Torontonians love to eat out, whether it's sitting at sidewalk bistros on a warm summer night or getting all bundled up for some hot Vietnamese pho.
# 
# So, I would love to explore a new investment and business in food. If we think of both city residents and travelers, they may want to choose different types of food. At the same time, they may want to choose local, district or signature Toronto's dishes.
# 
# In our business problem, we will perform analysis to understand the current market and decide which market and types of food that we will invest, and create a map with clusters to reveal which areas we should consider to start our business.
# 
# # Data
# 1) Table of postal codes in Toronto will be used to identify the Postal Code, Borough and Neighborhood.
# 2) Geospatial Data of Canada will be used in order to get the latitude and longitude coordinates of a given postal code in Canada as well as Toronto.
# 3) Foursquare API data will be used to explore venues around Toronto, the top 5 venues around these areas, explore users' tips, and finally perform clustering.

# In[528]:


import numpy as np # library for vectorized computation
import pandas as pd # library to process data as dataframes


# In[529]:


# extract data from downloaded postal information
df = pd.read_csv(r'C:\Users\daryle\Desktop\Canada.csv', header = None)

# assign column names
df.columns = ['Postcode', 'Borough', 'Neighborhood']

# Ignore cells with a borough that is Not assigned
df = df[~df['Borough'].isin(['Not assigned'])]

# Rename neighborhood by borough if Not assigned neighborhood
for i in range(df.shape[0]):
    if (df.iloc[i,2])=='Not assigned':
        df.iloc[i,2]=df.iloc[i,1]

# Groupby "Postcode" into one row, and join the Neighbourhood cell togehter with a comma
df=df.groupby(['Postcode', 'Borough'])['Neighborhood'].apply(', '.join).reset_index()

# Return shape
df.head()


# In[530]:


df_coor = pd.read_csv(r'C:\Users\daryle\Desktop\Geospatial_Coordinates.csv')
df_coor.head()


# In[531]:


df_merged = pd.merge(left=df,right=df_coor, how='left', left_on='Postcode', right_on='Postal Code')
df_merged=df_merged.drop(['Postal Code'], axis=1)

df_merged.head()


# In[532]:


df_merged_Toronto= df_merged[df_merged.Borough.str.contains('Toronto',case=False)].reset_index()

df_temp=df_merged_Toronto.groupby(['Borough']).agg(['count'])

df_temp


# In[533]:


#!pip install geopy
#!pip install folium


# In[534]:


from geopy.geocoders import Nominatim
address = 'Toronto, Ontario'
geolocator = Nominatim(user_agent="To_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# In[535]:


import folium

# create map of using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_merged_Toronto['Latitude'], df_merged_Toronto['Longitude'], df_merged_Toronto['Borough'], df_merged_Toronto['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[536]:


CLIENT_ID = 'P250FVYMNTWGMVNZJAZBDTVIOX2BPKYCA1WKH0PNZCKKQAAH' # your Foursquare ID
CLIENT_SECRET = 'SAYOGPVBHSQP53APNV2XSRM5XFJHQPAJKIT1TJJM3U3ZAOMS' # your Foursquare Secret
VERSION = '20180604'

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[537]:


df_merged_Toronto.loc[0, 'Neighborhood']


# In[538]:


neighborhood_latitude = df_merged_Toronto.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = df_merged_Toronto.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = df_merged_Toronto.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# In[539]:


import requests

LIMIT = 100
radius = 500 
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
results = requests.get(url).json()


# In[540]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[541]:


from pandas.io.json import json_normalize

venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[542]:


nearby_venues_visual=nearby_venues.groupby(['categories']).agg(['count'])
                                                                
nearby_venues_visual


# In[543]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[544]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[545]:


Toronto_venues = getNearbyVenues(names=df_merged_Toronto['Neighborhood'],
                                   latitudes=df_merged_Toronto['Latitude'],
                                   longitudes=df_merged_Toronto['Longitude']
                                     )
Toronto_venues.head()


# In[546]:


Toronto_venues_visual=Toronto_venues.groupby(['Venue Category']).agg(['count'])

Toronto_venues_visual


# In[547]:


Tor_ven= Toronto_venues[Toronto_venues['Venue Category'].str.contains('Restaurant',case=False)].reset_index()

indexNames = Tor_ven[Tor_ven['Venue Category'] == 'Restaurant' ].index
 
Tor_ven.drop(indexNames , inplace=True)

Tor_ven_visual=Tor_ven.groupby(['Venue Category']).agg(['count'])

Tor_ven_visual


# In[548]:


Toronto_venues=Tor_ven

# one hot encoding
Toronto_onehot = pd.get_dummies(Toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
Toronto_onehot['Neighborhood'] = Toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [Toronto_onehot.columns[-1]] + list(Toronto_onehot.columns[:-1])
Toronto_onehot = Toronto_onehot[fixed_columns]

Toronto_onehot.head()


# In[549]:


Toronto_onehot.shape


# In[550]:


Toronto_grouped = Toronto_onehot.groupby('Neighborhood').mean().reset_index()
Toronto_grouped


# In[551]:


Toronto_grouped.shape


# In[552]:


num_top_venues = 5

for hood in Toronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = Toronto_grouped[Toronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[553]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[554]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = Toronto_grouped['Neighborhood']

for ind in np.arange(Toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[555]:


Toronto_grouped.shape

path=r'C:\Users\daryle\Desktop\output.csv'

Toronto_grouped.to_csv(path)


# In[556]:


Toronto_grouped_dum=Toronto_grouped.drop('Neighborhood', 1)

Toronto_grouped_dum=Toronto_grouped_dum.T

Toronto_grouped_dum['Sum']=Toronto_grouped_dum.sum(axis = 1)

Toronto_grouped_dum=Toronto_grouped_dum.nlargest(20, 'Sum')

Top_3_Restaurant=Toronto_grouped_dum.nlargest(3, 'Sum')

Toronto_grouped_dum=Toronto_grouped_dum.drop(['Sum'], axis=1).T

Top_3_Restaurant


# In[557]:


from sklearn.cluster import KMeans

# set number of clusters
kclusters = 5

#Toronto_grouped_clustering = Toronto_grouped.drop('Neighborhood', 1)


Toronto_grouped_clustering = Toronto_grouped_dum


# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[558]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

Toronto_merged = df_merged_Toronto

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
Toronto_merged = Toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

Toronto_merged.head() # check the last columns!


# In[559]:


Toronto_merged=Toronto_merged.dropna()

Toronto_merged['Cluster Labels']=Toronto_merged['Cluster Labels'].astype(int)


# In[560]:


import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(Toronto_merged['Latitude'], Toronto_merged['Longitude'], Toronto_merged['Neighborhood'], Toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[566]:


Cluster_1=Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 0, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[567]:


Cluster_2=Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 1, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[568]:


Cluster_3=Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 2, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[569]:


Cluster_4=Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 3, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[570]:


Cluster_5=Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 4, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[571]:


path=r'C:\Users\daryle\Desktop\Cluster_1.csv'
Cluster_1.to_csv(path)

path=r'C:\Users\daryle\Desktop\Cluster_2.csv'
Cluster_2.to_csv(path)

path=r'C:\Users\daryle\Desktop\Cluster_3.csv'
Cluster_3.to_csv(path)

path=r'C:\Users\daryle\Desktop\Cluster_4.csv'
Cluster_4.to_csv(path)

path=r'C:\Users\daryle\Desktop\Cluster_5.csv'
Cluster_5.to_csv(path)

