from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XML, fromstring, tostring
import pandas as pd
import time

import osmnx as ox
import networkx as nx
import re

import multiprocessing as mp
import numpy as np
from joblib import Parallel, delayed

def speeds(G,API_KEY):
    nodes, edges = ox.graph_to_gdfs(G, fill_edge_geometry=False)
    y1 = str(nodes['y'].max())
    x1 = str(nodes['x'].min())
    y2 = str(nodes['y'].min())
    x2 = str(nodes['x'].max())

    page = requests.get(
        'https://traffic.api.here.com/traffic/6.2/flow.xml?app_id=' + API_KEY + '&bbox=' + y1 + ',' + x1 + ';' + y2 + ',' + x2 + '&responseattributes=sh,fc')

    soup = BeautifulSoup(page.text)
    roads = soup.find_all('fi')

    a1 = []
    loc_list_hv = []
    lats = []
    longs = []
    sus = []
    ffs = []
    c = 0
    for road in roads:
        # for j in range(0,len(shps)):
        myxml = fromstring(str(road))
        # fc=5
        for child in myxml:
            # print(child.tag, child.attrib)
            if ('fc' in child.attrib):
                fc = int(child.attrib['fc'])
            if ('cn' in child.attrib):
                cn = float(child.attrib['cn'])
            if ('su' in child.attrib):
                su = float(child.attrib['su'])
            if ('ff' in child.attrib):
                ff = float(child.attrib['ff'])
        # if((fc<=10) and (cn>=0.0)):
        shps = road.find_all("shp")
        for j in range(0, len(shps)):
            latlong = shps[j].text.replace(',', ' ').split()
            # loc_list=[]
            la = []
            lo = []
            su1 = []
            ff1 = []

            for i in range(0, int(len(latlong) / 2)):
                loc_list_hv.append([float(latlong[2 * i]), float(latlong[2 * i + 1]), float(su), float(ff)])
                la.append(float(latlong[2 * i]))
                lo.append(float(latlong[2 * i + 1]))
                su1.append(float(su))
                ff1.append(float(ff))
            lats.append(la)
            longs.append(lo)
            sus.append(np.mean(su1))
            ffs.append(np.mean(ff1))

    lats_r = []
    longs_r = []
    speeds_r = []
    speeds_f = []
    speeds_rat = []
    for i in range(0, len(lats)):
        for j in range(len(lats[i])):
            lats_r.append(lats[i][j])
            longs_r.append(longs[i][j])
            speeds_r.append(sus[i])
            speeds_f.append(ffs[i])
            speeds_rat.append(sus[i] / ffs[i])
        # print(i,len(lats))

    data = np.array([lats_r, longs_r, speeds_r, speeds_f, speeds_rat])
    df = pd.DataFrame(data=data[:, 1:].T, columns=["latitude", "longitude", "speeds", "speeds_f", "speeds_rat"])
    b=df.copy()
    b.drop(b.loc[b['speeds_rat'] < 0].index, inplace=True)
    b.drop_duplicates(subset=['latitude', 'longitude'], inplace=True)
    b = b.reset_index(drop=True)
    b.to_csv('df_speeds.csv')

    return(b)


def edges_r(i,b,G):
    # dicty=defaultdict(int)
    # for i in range(0,len(b)):
    X = b['longitude'][i]
    Y = b['latitude'][i]
    c, dist = ox.distance.nearest_edges(G, X, Y, return_dist=True)
    (u, v, key) = c
    # edges.loc[u,v]['speed']=b['speeds'][i]
    # dicty[u,v]+=1
    return (u, v, key, dist, b['speeds'][i],X,Y)

def add_edge_speeds3(G, edges,hwy_speeds=None, fallback=None, precision=1, agg=np.mean):
    """
    Add edge speeds (km per hour) to graph as new `speed_kph` edge attributes.
    By default, this imputes free-flow travel speeds for all edges via the
    mean `maxspeed` value of the edges of each highway type. For highway types
    in the graph that have no `maxspeed` value on any edge, it assigns the
    mean of all `maxspeed` values in graph.
    This default mean-imputation can obviously be imprecise, and the user can
    override it by passing in `hwy_speeds` and/or `fallback` arguments that
    correspond to local speed limit standards. The user can also specify a
    different aggregation function (such as the median) to impute missing
    values from the observed values.
    If edge `maxspeed` attribute has "mph" in it, value will automatically be
    converted from miles per hour to km per hour. Any other speed units should
    be manually converted to km per hour prior to running this function,
    otherwise there could be unexpected results. If "mph" does not appear in
    the edge's maxspeed attribute string, then function assumes kph, per OSM
    guidelines: https://wiki.openstreetmap.org/wiki/Map_Features/Units
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    hwy_speeds : dict
        dict keys = OSM highway types and values = typical speeds (km per
        hour) to assign to edges of that highway type for any edges missing
        speed data. Any edges with highway type not in `hwy_speeds` will be
        assigned the mean preexisting speed value of all edges of that highway
        type.
    fallback : numeric
        default speed value (km per hour) to assign to edges whose highway
        type did not appear in `hwy_speeds` and had no preexisting speed
        values on any edge
    precision : int
        decimal precision to round speed_kph
    agg : function
        aggregation function to impute missing values from observed values.
        the default is numpy.mean, but you might also consider for example
        numpy.median, numpy.nanmedian, or your own custom function
    Returns
    -------
    G : networkx.MultiDiGraph
        graph with speed_kph attributes on all edges
    """
    if fallback is None:
        fallback = np.nan

    #edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=False)

    # collapse any highway lists (can happen during graph simplification)
    # into string values simply by keeping just the first element of the list
    edges["highway"] = edges["highway"].map(lambda x: x[0] if isinstance(x, list) else x)

    if "speed" in edges.columns:
        # collapse any maxspeed lists (can happen during graph simplification)
        # into a single value
        edges["speed"] = edges["speed"].apply(_collapse_multiple_maxspeed_values, agg=agg)

        # create speed_kph by cleaning maxspeed strings and converting mph to
        # kph if necessary
        edges["speed_kph"] = edges["speed"].astype(str).map(_clean_maxspeed).astype(float)
    else:
        # if no edges in graph had a maxspeed attribute
        edges["speed_kph"] = None

    # if user provided hwy_speeds, use them as default values, otherwise
    # initialize an empty series to populate with values
    if hwy_speeds is None:
        hwy_speed_avg = pd.Series(dtype=float)
    else:
        hwy_speed_avg = pd.Series(hwy_speeds).dropna()

    # for each highway type that caller did not provide in hwy_speeds, impute
    # speed of type by taking the mean of the preexisting speed values of that
    # highway type
    for hwy, group in edges.groupby("highway"):
        if hwy not in hwy_speed_avg:
            hwy_speed_avg.loc[hwy] = agg(group["speed_kph"])

    # if any highway types had no preexisting speed values, impute their speed
    # with fallback value provided by caller. if fallback=np.nan, impute speed
    # as the mean speed of all highway types that did have preexisting values
    hwy_speed_avg = hwy_speed_avg.fillna(fallback).fillna(agg(hwy_speed_avg))

    # for each edge missing speed data, assign it the imputed value for its
    # highway type
    speed_kph = (
        edges[["highway", "speed_kph"]].set_index("highway").iloc[:, 0].fillna(hwy_speed_avg)
    )

    # all speeds will be null if edges had no preexisting maxspeed data and
    # caller did not pass in hwy_speeds or fallback arguments
    if pd.isnull(speed_kph).all():
        raise ValueError(
            (
                "this graph's edges have no preexisting `maxspeed` "
                "attribute values so you must pass `hwy_speeds` or "
                "`fallback` arguments."
            )
        )

    # add speed kph attribute to graph edges
    edges["speed_kph"] = speed_kph.round(precision).values
    nx.set_edge_attributes(G, values=edges["speed_kph"], name="speed_kph")

    return G


def _clean_maxspeed(value, convert_mph=True):
    """
    Clean a maxspeed string and convert mph to kph if necessary.
    Parameters
    ----------
    value : string
        an OSM way maxspeed value
    convert_mph : bool
        if True, convert mph to kph
    Returns
    -------
    value_clean : string
    """
    MPH_TO_KPH = 1.60934
    pattern = re.compile(r"[^\d\.,;]")

    try:
        # strip out everything but numbers, periods, commas, semicolons
        value_clean = float(re.sub(pattern, "", value).replace(",", "."))
        if convert_mph and "mph" in value.lower():
            value_clean = value_clean * MPH_TO_KPH
        return value_clean

    except ValueError:
        return None


def _collapse_multiple_maxspeed_values(value, agg):
    """
    Collapse a list of maxspeed values to a single value.
    Parameters
    ----------
    value : list or string
        an OSM way maxspeed value, or a list of them
    agg : function
        the aggregation function to reduce the list to a single value
    Returns
    -------
    agg_value : int
        an integer representation of the aggregated value in the list,
        converted to kph if original value was in mph.
    """
    # if this isn't a list, just return it right back to the caller
    if not isinstance(value, list):
        return value

    else:
        try:
            # clean each value in list and convert to kph if it is mph then
            # return a single aggregated value
            values = [_clean_maxspeed(x) for x in value]
            return int(agg(pd.Series(values).dropna()))
        except ValueError:
            return None


def get_edge_speeds(G,API_KEY):
    b=speeds(G, API_KEY)
    edges_speeds = Parallel(n_jobs=-1, verbose=10)([delayed(edges_r)(i,b,G) for i in range(len(b))])

    df_es = pd.DataFrame(edges_speeds)
    df_es.to_csv('df_es.csv')


def add_edge_speeds(G,df_es,cut):

    df_es = df_es[df_es['3'] <= cut]  # making sure no points too far from OX graph are there - this seems like a good threshold from visualization

    df_es = df_es[['0', '1', '2', '4']]
    df_es.columns = ['u', 'v', 'key', 'speed']
    df_es = df_es.groupby(['u', 'v', 'key']).mean().reset_index()
    df_es.index = pd.MultiIndex.from_frame(df_es[['u', 'v', 'key']])
    df_es.columns = ['u1', 'v1', 'key1', 'speed']
    nodes, edges = ox.graph_to_gdfs(G, fill_edge_geometry=False)
    edges2 = edges.merge(df_es, left_index=True, right_index=True, how='left')

    # now adding travel times backed by HERE traffic data for subsequent processing
    G1 = add_edge_speeds3(G, edges2)
    G1 = ox.add_edge_travel_times(G1)
    edgesG1 = ox.graph_to_gdfs(G1, nodes=False, fill_edge_geometry=False)

    return(edgesG1)



