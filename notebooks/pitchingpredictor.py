
# standard imports
import numpy as np
import os

# scraping tools
import requests
from bs4 import BeautifulSoup

# data management tools
import pandas as pd

# data analysis toolz
import scipy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


    
def scrape_year(year='2019',cat='pit',verbose=0):
    '''
    
    scrape the total, season-long player data

    inputs
    ---------
    year    : string of the year to query
    cat     : 'bat' or 'pit' for batting or pitching statistics
    verbose : level of reporting. 0=none.
    
    returns
    ---------
    PDict
    NDict
    NDict2
    TDict
    
    todo
    ---------
    -change internal nomenclature to be sensical
    -check table query number
    
    
    '''
    
    year_url      =   "https://www.fangraphs.com/leaders.aspx?pos=all&stats="+str(cat)+\
                      "&lg=all&qual=0&type=0&season="+str(year)+\
                      "&month=0&season1="+str(year)+\
                      "&ind=0&team=0&rost=0&age=0&filter=&players=0&page=1_1200"

    if verbose: print('The year is {}'.format(year))
        
    # perform the lookup
    r               = requests.get(year_url)

    # old compatibility version: save which checking
    #soup            = BeautifulSoup(r.content, "html5lib")
    
    soup            = BeautifulSoup(r.text, "html5lib")

    # identify the master table
    table_data      = soup.find("table", { "class" : "rgMasterTable"})      
                
    # populate header values
    headers = [header.text for header in table_data.findAll('th')]
    
    # cycle through all rows in the table
    rows = []
    for row in table_data.findAll("tr"):
        cells = row.findAll("td")
        if len(cells) != 0:
            for td in row.findAll("td"):       
                sav2 = [td.getText() for td in row.findAll("td")] 
            rows.append(sav2)
    
    # transform to a pandas table
    df = pd.DataFrame(rows, columns=headers)
    df['Year'] = year
    
    # return the dataframe
    return df
   



def compute_cluster_pitching(df,years,nclusters,min_ip=10,verbose=0):
    '''
    find the (pitching) cluster centers
    
    inputs
    -----------
    df        : dataframe in the format given by scrape_year
    years     : list of years to use
    nclusters : number of clusters to fit
    min_ip    : (default=10) minimum number of innings pitching
    verbose   : (default=0) level of reporting
    
    returns
    -----------
    year_df             :
    df                  :
    stereotype_df       :
    cluster_centroid_df :
    
    
    todo
    -----------
    -change internal naming conventions to make sense
    
    '''


    df = df.loc[df['Name']==df['Name'] ]

    for column in df.columns[3:]:
        if column != 'wRC+':
            try:
                df[column] = df[column].astype(float)
            except:
                df[column] = df[column].str[:-1]
                df[column] = df[column].astype(float)

    # if selecting for starters
    #df = df.loc[(df['TBF']> 150) & (df['GS'] > 10) ]

    # if selecting for relievers
    #df = df.loc[(df['TBF']> 100) & (df['GS'] < 10) ]
    
    # if selecting for starters
    df = df.loc[(df['TBF']> 150)]



    fantasy_stats = ['HR', 'ER', 'BB', 'H', 'SO']
    
    denominator = 'TBF'
    denominator = 'IP'


    for stat in fantasy_stats:
        if stat=='SO':
            df['{0}.Normalize'.format(stat)] = 100.*df[stat]/df[denominator]
        elif stat=='H':
            df['{0}.Normalize'.format(stat)] = -100.*(df[stat]-df['HR'])/df[denominator]
        elif (stat != 'W') & (stat != 'SV'):
            #df['{0}.Normalize'.format(stat)] = df[stat]*200.0/df['IP']
            df['{0}.Normalize'.format(stat)] = -100.*df[stat]/df[denominator]
        else:
            df['{0}.Normalize'.format(stat)] = 10.*df[stat]/df['G']

    Y = df[['HR.Normalize','ER.Normalize',\
        'BB.Normalize','H.Normalize', 'SO.Normalize']].values


    # HR, ER, BB, H, SO
    kmeans = KMeans(n_clusters=nclusters, random_state=3425)
    
    kmeans.fit(Y)
    predict = kmeans.predict(Y)
    transform = kmeans.transform(Y)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # set up the cluster centroid dataframe
    cluster_centroid_df = pd.DataFrame(centroids, columns=['HR.Centroid', 'ER.Centroid', 'BB.Centroid', 'H.Centroid', 'SO.Centroid'])

    cluster_centroid_df['Tot.Rank']  = 0
    for column in cluster_centroid_df.columns[:-1]:
        stat = column.split(".")[0]
        
        # this formula scales relative to overall value
        meanval = np.nanmedian(df['{0}.Normalize'.format(stat)])
        stdval = np.nanstd(df['{0}.Normalize'.format(stat)])

        cluster_centroid_df['{0}.Rank'.format(stat)]  = (cluster_centroid_df['{0}.Centroid'.format(stat)] - meanval)/stdval
        cluster_centroid_df['Tot.Rank'] = cluster_centroid_df['Tot.Rank'] +cluster_centroid_df['{0}.Rank'.format(stat)]


    # predict the clusters
    df['Clusters'] = pd.Series(predict, index=df.index)

    short_pitcher_df = df.loc[df[denominator]>min_ip]

    short_pitcher_df = short_pitcher_df[['Name', 'Year', 'HR.Normalize','ER.Normalize', 'BB.Normalize','H.Normalize', 'SO.Normalize','Clusters']]

    # Merge in the centroids for comparisons and find the sum of each stat's deviation for each player.
    short_pitcher_df = short_pitcher_df.merge(cluster_centroid_df, right_index = True, left_on='Clusters')
    short_pitcher_df['Centroid Diff'] = 0
    pitch_fields = ['HR.{0}','ER.{0}', 'BB.{0}','H.{0}', 'SO.{0}']
    for field in pitch_fields:
        
        #
        short_pitcher_df[field.format('Diff')] = np.abs(short_pitcher_df[field.format('Centroid')] - short_pitcher_df[field.format('Normalize')])
        
        #
        short_pitcher_df[field.format('Diff')] = short_pitcher_df[field.format('Diff')].rank(pct=True)
        
        #
        short_pitcher_df['Centroid Diff'] = short_pitcher_df['Centroid Diff'] + short_pitcher_df[field.format('Diff')]
    
    # Now we can use the deviation sums to find the player closest to each centroid and create a DataFrame of stereotype players.
    idx = short_pitcher_df.groupby(['Clusters'])['Centroid Diff'].transform(min) == short_pitcher_df['Centroid Diff']
    stereotype_df = short_pitcher_df[idx].copy()

    # Clean up the DataFrame...
    for stat in fantasy_stats:
        stereotype_df['{0}.Normalize'.format(stat)] = stereotype_df['{0}.Normalize'.format(stat)] - stereotype_df['{0}.Normalize'.format(stat)] % 1
        
    stereotype_df = stereotype_df[['Clusters', 'Name', 'Year', 'HR.Normalize', 'ER.Normalize', 'BB.Normalize', 'H.Normalize', 'SO.Normalize']]

    
    # Update cluster values
    cluster_centroid_df['Value Cluster'] = cluster_centroid_df['Tot.Rank'].rank(ascending      =1, method = 'first') 
    cluster_centroid_df['Clusters'] = cluster_centroid_df.index
    cluster_equiv = cluster_centroid_df[['Clusters', 'Value Cluster']]
    stereotype_df = stereotype_df.merge(cluster_equiv, on = 'Clusters', how = 'left')

    
    stereotype_df = stereotype_df.sort_values(['Value Cluster'], ascending = True)


    df = df.merge(cluster_equiv, on = 'Clusters', how = 'left')

    year_dfs2 = []
    for year in years:
        year_df = df.loc[df['Year'] == year]
        year_df = year_df[['Name', 'Value Cluster']]
        year_df.columns = ['Name', 'Value Cluster {0}'.format(year)]
        year_dfs2.append(year_df)

    year_df = year_dfs2[0]

    for i in year_dfs2[1:]:
        year_df = year_df.merge(i, on = 'Name', how = 'outer')
        # clean dataframe by filling nulls with zeros

    for column in year_df.columns[1:]:
        year_df[column] = year_df[column].fillna(0)
    
    
    year_df.to_csv('../tables/Clusters_By_Year_starters{}.csv'.format(nclusters), index = False)
    df.to_csv('../tables/All_Player_Data_starters{}.csv'.format(nclusters), index = False)
    stereotype_df.to_csv('../tables/Stereotype_Players_starters{}.csv'.format(nclusters), index = False)


    return year_df,df,stereotype_df,cluster_centroid_df,transform






def generate_player_prediction(pl,df,cluster_centroid_df,\
                               estimated_ips=600,\
                               year_weights=[1.,1.,1.,1.],\
                               year_weights_penalty=[0.,0.,0.,0.],\
                               regression_factor=1.,err_regression_factor=1.,\
                               AgeDict={},verbose=0):
    '''
    generate predictions for an individual player
    
    
    inputs
    --------
    pl                         : (string) the name of the player
    df                         : the full data table
    cluster_centroid_df        : the centroid table
    estimated_pas              : (default=600) the estimate number of plate appearances 
    year_weights               : the weights for individual years
    year_weights_penalty       : the weighting penalty for missing a year
    regression_factor          : the regression factor to the cluster center
    err_regression_factor      : the regression factor for error estimation
    AgeDict                    : dictionary of ages
    verbose                    : (default=0) 
    
    returns
    --------
    
    
    todo
    -------
    -modularlize age penalty
    
    
    '''
    # initialize the blank arrays
    pstats = np.zeros(6)
    perr = np.zeros(6)
    
    # initialize counters
    yrsum = 0.
    yrsum_denom = 0.
    nyrs = 0.
    ip_s = 0.
    ab = -1.
    
    # which stats are we predicting?
    fantasy_stats = ['HR', 'ER', 'BB', 'H', 'SO']



    try:
        age = float(AgeDict[pl])
    except:
        pass
        if verbose > 1: print('No age for', pl)
        
        # default to no penalties
        age = 25.0
        
        
    # find all available years for data
    available_years = df['Year'][df['Name']==pl]

    for year in available_years:

        # if year is available, analyze
        try:
            
            # identify the value cluster for the year 
            v = list(df['Value Cluster'][(df['Name']==pl) & (df['Year']==year) ])[0]

            # loop through the desired stats for predictions
            # look up the comparison value in the centroid table
            for indx,stat in enumerate(fantasy_stats):      
                statcen = list(cluster_centroid_df['{0}.Centroid'.format(stat)]\
                      [cluster_centroid_df['Value Cluster']==v])[0]
                statnorm = list(df['{0}.Normalize'.format(stat)][(df['Name']==pl) & (df['Year']==year) ])[0]

                #print(year,stat,np.round(statcen,2),np.round(statnorm,2),year_weights[year])
                
                # set the regression to the cluster center
                # this can be modularized to take asymmetric data
                pstats[indx] += year_weights[year] * ( regression_factor*(statnorm-statcen) + statcen)
                
                # these could be added in quadrature for more accuracy, just have to watch the weights
                perr[indx] += year_weights[year] * ( err_regression_factor*np.abs(statnorm-statcen))
            
            # keep track of the denominators
            yrsum += year_weights[year]
            yrsum_denom += year_weights[year]
            
            
            # apply the age factor
            #                  0.1          * (33 - 33)        + (-1)
            #age_penalty = age_penalty_slope * ((age-age_pivot) + (year-2019.))
            #print(year,age_penalty)
            #yrsum -= np.max([age_penalty,0.0])
            
            # increment counted years
            nyrs += 1.
        
        # if the year doesn't exist, apply a penalty to the denominator
        except:
            yrsum_denom -= year_weights_penalty[year]
        
    
    # zero out if no available data
    if nyrs < 0.5: nyrs=1000.

    # concatenate the data for printing
    if estimated_ips > 10:
        print(pl,end=', ')
        for indx,p in enumerate(pstats):
            print(np.abs(np.round(estimated_ips*p/100.,2)),end=', ')
            print(np.round(estimated_ips*perr[indx]/100.,2),end=', ')

        # for computing adjustment factors later...
        print(int(estimated_ips),end=', ')
        print(np.round(yrsum/yrsum_denom,2),end=', ')

        print('')





