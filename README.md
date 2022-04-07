# osmnx-edge-speeds
Obtaining road speeds corresponding to OSMnx urban road networks from the HERE traffic API

# Usage:

## 1. Create a HERE free API account and get API credentials:

![image](https://user-images.githubusercontent.com/29293526/162217886-17521122-f014-4688-862a-8a580ec4595e.png)

## 2. obtain the required map from OSMnx:

place_name="Washington DC, USA"
G = ox.graph_from_place(place_name,network_type='drive', simplify=True)

## 3. Run get_egge_speeds from the get_speeds.py file -make sure to replace the API_KEY with your key when you created your account

get_edge_speeds(G,API_KEY)

## 4. Obtain edges by running add_edge_speeds from the get_speeds.py file. choose a cutoff distance (0.01 here) corresponding to the maximum distance between OSMnx nodes and HERE lat-long data

df_es=pd.read_csv('df_es.csv')
edges1=add_edge_speeds(G,df_es,0.01)


## And.. that's it! You now have your OSMnx edge files with speeds!


