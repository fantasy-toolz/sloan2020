
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


    
def scrape_year(year='2019',cat='bat',verbose=0):
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
   


def compute_cluster(df,years,nclusters,min_pas=150,verbose=0):
    '''
    find the cluster centers
    
    inputs
    -----------
    df        : dataframe in the format given by scrape_year
    years     : list of years to use
    nclusters : number of clusters to fit
    min_pas   : (default=150) minimum number of plate appearances
    verbose   : (default=0) level of reporting
    
    returns
    -----------
    year_df                    :
    df                         :
    stereotype_df              :
    hitter_cluster_centroid_df :
    transform                  :
    
    
    todo
    -----------
    -change internal naming conventions to make sense
    -needs a silent version
    
    '''


    df = df.loc[df['Name']==df['Name'] ]#) & (hitter_eoy_df['wRC+'] != '&nbsp;')]

    # verify that we are selecting the appropriate years
    df = df.loc[df['Year'].isin(years)]
    
    for column in df.columns[3:]:
        if column != 'wRC+':
            try:
                df[column] = df[column].astype(float)
            except:
                df[column] = df[column].str[:-1]
                df[column] = df[column].astype(float)

    # put in a limit for minimum plate appearances
    df = df.loc[(df['PA']> min_pas)]

    # select the categories to include
    fantasy_stats = ['HR', 'H', 'AB', 'SB', 'RBI','R']
    
    denominator = 'PA'


    # normalize stats by plate appearances
    for stat in fantasy_stats:
        
        # for hits, remove the HRs
        if stat=='H':
            df['{0}.Normalize'.format(stat)] = 100.*(df[stat]-df['HR'])/df[denominator]
            
        else:
            df['{0}.Normalize'.format(stat)] = 100.*(df[stat])/df[denominator]


    # generate the data frame for clustering
    Y = df[['HR.Normalize','H.Normalize',\
        'AB.Normalize','SB.Normalize', 'RBI.Normalize', 'R.Normalize']].values



    # HR, H, AB, SB, RBI, R
    kmeans = KMeans(n_clusters=nclusters, random_state=3425)
    
    kmeans.fit(Y)
    predict = kmeans.predict(Y)
    transform = kmeans.transform(Y)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    #print(centroids)
    
    
    hitter_cluster_centroid_df = pd.DataFrame(centroids, columns=['HR.Centroid', 'H.Centroid', 'AB.Centroid', 'SB.Centroid', 'RBI.Centroid', 'R.Centroid'])

    hitter_cluster_centroid_df['Tot.Rank']  = 0
    for column in hitter_cluster_centroid_df.columns[:-1]:
        stat = column.split(".")[0]
        # this formula scales relative to overall value
        meanval = np.nanmean(df['{0}.Normalize'.format(stat)])
        stdval = np.nanstd(df['{0}.Normalize'.format(stat)])

        if stat != 'AB':
            hitter_cluster_centroid_df['{0}.Rank'.format(stat)]  = (hitter_cluster_centroid_df['{0}.Centroid'.format(stat)] - meanval)/stdval
            hitter_cluster_centroid_df['Tot.Rank'] = hitter_cluster_centroid_df['Tot.Rank'] +hitter_cluster_centroid_df['{0}.Rank'.format(stat)]


    # predict the clusters
    df['Clusters'] = pd.Series(predict, index=df.index)

    short_hitter_df = df.loc[df['PA']>min_pas]

    # simplify this DataFrame.
    short_hitter_df = short_hitter_df[['Name', 'Year', 'HR.Normalize','H.Normalize', 'AB.Normalize','SB.Normalize', 'RBI.Normalize', 'R.Normalize','Clusters', 'HR','H', 'AB','SB', 'RBI', 'R',]]

    # Merge in the centroids for comparisons and find the sum of each stat's deviation for each player.
    short_hitter_df = short_hitter_df.merge(hitter_cluster_centroid_df, right_index = True, left_on='Clusters')
    short_hitter_df['Centroid Diff'] = 0
    hit_fields = ['HR.{0}','H.{0}', 'AB.{0}','SB.{0}', 'RBI.{0}', 'R.{0}']
    for field in hit_fields:
        short_hitter_df[field.format('Diff')] = abs(short_hitter_df[field.format('Centroid')] - short_hitter_df[field.format('Normalize')])
        short_hitter_df[field.format('Diff')] = short_hitter_df[field.format('Diff')].rank(pct=True)
        short_hitter_df['Centroid Diff'] = short_hitter_df['Centroid Diff'] + short_hitter_df[field.format('Diff')]
    
    # Now we can use the deviation sums to find the player closest to each centroid and create a DataFrame of stereotype players.
    idx = short_hitter_df.groupby(['Clusters'])['Centroid Diff'].transform(min) == short_hitter_df['Centroid Diff']
    stereotype_df = short_hitter_df[idx].copy()

    # cleaning...
    for stat in fantasy_stats:
        stereotype_df['{0}.Normalize'.format(stat)] = stereotype_df['{0}.Normalize'.format(stat)] - stereotype_df['{0}.Normalize'.format(stat)] % 1
        stereotype_df['{0}'.format(stat)] = stereotype_df['{0}'.format(stat)]
        #stereotype_df['BA.Normalize'] = stereotype_df['H.Normalize'] / 600
    
    # decide to include normalized values?
    #stereotype_df = stereotype_df[['Clusters', 'Name', 'Year', 'HR.Normalize', 'H.Normalize', 'AB.Normalize', 'SB.Normalize', 'RBI.Normalize', 'R.Normalize']]
    
    # or the real values?
    stereotype_df = stereotype_df[['Clusters', 'Name', 'Year', 'HR', 'H', 'AB', 'SB', 'RBI', 'R']]
    

    
    # Update cluster values
    hitter_cluster_centroid_df['Value Cluster'] = hitter_cluster_centroid_df['Tot.Rank'].rank(ascending      =1, method = 'first') 
    hitter_cluster_centroid_df['Clusters'] = hitter_cluster_centroid_df.index
    cluster_equiv = hitter_cluster_centroid_df[['Clusters', 'Value Cluster']]
    stereotype_df = stereotype_df.merge(cluster_equiv, on = 'Clusters', how = 'left')
    stereotype_df = stereotype_df.sort_values(['Value Cluster'], ascending = False)


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
        # Clean DataFrame by filling nulls with zeros

    for column in year_df.columns[1:]:
        year_df[column] = year_df[column].fillna(0)
    
    year_df.to_csv('../tables/2019Clusters_By_Year_starters{}.csv'.format(nclusters), index = False)
    df.to_csv('../tables/2019All_Player_Data_starters{}.csv'.format(nclusters), index = False)
    stereotype_df.to_csv('../tables/2019Stereotype_Players_starters{}.csv'.format(nclusters), index = False)
    
    hitter_cluster_centroid_df = hitter_cluster_centroid_df.sort_values(['Value Cluster'], ascending = False)

    return year_df,df,stereotype_df,hitter_cluster_centroid_df,transform




def generate_player_prediction(pl,df,hitter_cluster_centroid_df,\
                               estimated_pas=600,\
                               year_weights=[1.,1.,1.,1.],\
                               year_weights_penalty=[0.,0.,0.,0.],\
                               regression_factor=1.,err_regression_factor=1.,\
                               AgeDict={},verbose=0,return_stats=False):
    '''
    generate predictions for an individual player
    
    
    inputs
    --------
    pl                         : (string) the name of the player
    df                         : the full data table
    hitter_cluster_centroid_df : the centroid table
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
    fantasy_stats = ['HR', 'H', 'AB', 'SB', 'RBI','R']



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
                statcen = list(hitter_cluster_centroid_df['{0}.Centroid'.format(stat)]\
                      [hitter_cluster_centroid_df['Value Cluster']==v])[0]
                statnorm = list(df['{0}.Normalize'.format(stat)][(df['Name']==pl) & (df['Year']==year) ])[0]

                #print(year,stat,np.round(statcen,2),np.round(statnorm,2),year_weights[year])
                
                # set the regression to the cluster center
                # this can be modularized to take asymmetric data
                pstats[indx] += year_weights[year] * ( regression_factor*(statnorm-statcen) + statcen)
                
                # these could be added in quadrature for more accuracy, just have to watch the weights
                
                #print(df[stat])
                # add in the Possion error as a minimum uncertainty
                statnormraw = list(df['{0}'.format(stat)][(df['Name']==pl) & (df['Year']==year) ])[0]
                normraw = list(df['PA'][(df['Name']==pl) & (df['Year']==year) ])[0]
                # alternate, or additional, would be to have the mean distance between clusters (e.g. what would happen if you were assigned to the incorrect cluster).
    
                statsdiff = np.nanmin([err_regression_factor*np.abs(statnorm-statcen),\
                                      100.*np.sqrt(statnormraw)/normraw])
                
                perr[indx] += year_weights[year] * ( statsdiff )
            
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
    Stats = {}
    Stats['PA'] = estimated_pas
    if estimated_pas > 200:
        if verbose: print(pl,end=', ')
        for indx,p in enumerate(pstats):
            Stats[fantasy_stats[indx]] = np.round(estimated_pas*p/100.,2)
            Stats['e'+fantasy_stats[indx]] = np.round(estimated_pas*perr[indx]/100.,2)

            if verbose:
                print(np.round(estimated_pas*p/100.,2),end=', ')
                print(np.round(estimated_pas*perr[indx]/100.,2),end=', ')

        # for computing adjustment factors later...
        if verbose:
            print(int(estimated_pas),end=', ')
            print(np.round(yrsum/yrsum_denom,2),end=', ')

            print('')


    if return_stats:
        return Stats





def make_manual_predictions_batting(lght_df,hitter_cluster_centroid_df,chkstat,\
                                    year_weights=[0.05,0.05,0.3,0.6],\
                                    prefac=0.5,prefac_err=1.,nclusters=12,verbose=0):
    """
    make a prediction through the years for a single player
    
    inputs
    -------------------
    lght_df
    df
    hitter_cluster_centroid_df
    chkstat
    year_weights=[0.05,0.05,0.3,0.6]
    prefac=0.5
    prefac_err=1.
    nclusters=12.
    
    
    
    """
    
    # how many years of data do we have?
    uni_years = np.unique(lght_df['Year'])
    clustervals = np.zeros([uni_years.size,nclusters])


    
    indi_values = np.zeros(uni_years.size)
    unc_values = np.zeros(uni_years.size)

    # assign each year to a cluster
    for iyear,year in enumerate(uni_years):
        for val in lght_df.loc[lght_df['Year']==year][chkstat+'.Normalize']:
            clustervals[iyear] += (np.abs(val-hitter_cluster_centroid_df[chkstat+'.Centroid']))
    

    # cycle through the years for the predictions
    for iyear,year in enumerate(uni_years):
        bestcluster = np.argmin(clustervals[iyear])

        bestclusterval = np.array(hitter_cluster_centroid_df['Value Cluster'])[bestcluster]

        
        lght_df_year = lght_df.loc[lght_df['Year']==year]
        statval = np.array(lght_df_year[chkstat+'.Normalize'])[0]

        cent_df = hitter_cluster_centroid_df.loc[hitter_cluster_centroid_df['Value Cluster']==bestclusterval]
        statval_predict = np.array(cent_df[chkstat+'.Centroid'])[0]

        
        # make the prediction from the year
        indi_values[iyear] = statval_predict+prefac*(statval-statval_predict)

        # make the error prediction from the year
        unc_values[iyear] = prefac_err*(statval-statval_predict)

        # compute the Poissonian rate for the minimum uncertainty
        poiss_rate = 100.*np.array(np.sqrt(lght_df_year[chkstat])/lght_df_year['PA'])[0]

        unc_values[iyear] = np.nanmax([prefac_err*(statval-statval_predict),poiss_rate])


    pred_vals = np.zeros(uni_years.size)
    pred_errs = np.zeros(uni_years.size)
    pred_sum = np.zeros(uni_years.size)


    # weight the years appropriately to come up with the total predictions
    for iweight,weight in enumerate(year_weights[::-1]):
        for indx in range(0,uni_years.size):
            if indx-iweight >= 0:
                pred_vals[indx] += weight*indi_values[indx-iweight]
                pred_errs[indx] += weight*unc_values[indx-iweight]
                pred_sum[indx] += weight

    # normalize the prediction years
    pred_vals /= pred_sum

    if verbose:
        print('Predicted values:',pred_vals)
        print('Predicted errors:',pred_errs)

    return uni_years,indi_values,unc_values,pred_vals,pred_errs


