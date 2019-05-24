import pandas as pd
import numpy as np
from sklearn import preprocessing



def data_cleaning(filename,normalize=True):

    main = pd.read_csv (filename,encoding = 'latin-1') 

    #Deleting the space in the column name
    main.columns = main.columns.str.replace(' ', '')

    #Remove duplicated rows if any
    main.drop_duplicates(inplace = True)

    #We see that PID and Order might not be usefull for getting prediction as they only a unique ID and sequence number
    #that does not have correlation to SalePrice or to other columns
    main = main.drop(['PID','Order'], 1)

    # List of the column name which have NAN value, but it is actually and NA (no facility available)
    # we fill in the missing value with string "NA" since it is not missing value, so we can score them properly
    # To score them, we groupby by the column name and categorize them by the mean of sale price for each unique value
    # sort the unique value by the mean of sale price from low to high and keep them in the list
    # for each value in the columns, replace with the index from the list(which shows the rank/score), for each value.
    no_facility = ['Alley','FireplaceQu','GarageFinish','GarageType','GarageQual','GarageCond','PoolQC','Fence','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']

    for name in no_facility:
        main[name]=main[name].fillna('NA')
        grouped = main.groupby(name)['SalePrice'].mean().reset_index().sort_values(by='SalePrice').reset_index(drop=True)
        replace_list = list(grouped[name])
        main[name]=main[name].apply(lambda x: replace_list.index(x))


    #Fill in MiscFeature with NA, since NAN is Not available
    main['MiscFeature']=main['MiscFeature'].fillna('NA')

    #MasVnrType and MasVnrArea
    #fill in with None for the NA, as one of the category of Mas VnrType is None, and for category NOne, the MasVnrArea is 0
    main['MasVnrType']=main['MasVnrType'].fillna('None')
    main['MasVnrArea']=main['MasVnrArea'].fillna(0)

    #There is inconsistency with the MasVnrType, None, some of it have value > 0
    temp_df=main[main.MasVnrType=='None']
    MasVnrNone = temp_df.loc[:,['MasVnrType','MasVnrArea']].pivot_table(index=['MasVnrType','MasVnrArea'],aggfunc='count')

    #as wee see that None should have value 0, for the one have value 1 is possbily typo, so we replace with 0
    main.loc[(main.loc[:,'MasVnrArea'] == 1) & (main.loc[:,'MasVnrType'] == 'None'),'MasVnrArea'] = 0

    #For MasVnrType None, with MasVnrArea > 0, we change the MasVnrType to the mode() of the MasVnrType (as it should not be NONE)
    main.loc[(main.loc[:,'MasVnrArea'] != 0) & (main.loc[:,'MasVnrType'] == 'None'),'MasVnrType'] =main.loc[main.loc[:,'MasVnrType'] != 'None','MasVnrType'].mode()[0]

    ##Fill in GarageCars and Garage Area with 0, as the status of GarageType is NA
    main['GarageCars']=main['GarageCars'].fillna(0)
    main['GarageArea']=main['GarageArea'].fillna(0)

    #convert categorical to numerical manually for columns with ranking 
    cleanup_dicts= {'ExterQual':    {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa':1, 'Po':0},
                'ExterCond':    {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa':1, 'Po':0},
                'Functional' :  {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod':4, 'Maj1':3, 'Maj2':2,'Sev': 1, 'Sal': 0},
                'SaleCondition':{'Normal': 5, 'Abnorml':4, 'AdjLand':3,'Alloca':2,'Family':1,'Partial': 0},
                'HeatingQC':    {'Normal': 5, 'Abnorml':4, 'AdjLand':3,'Alloca':2,'Family':1,'Partial': 0},
                'LotShape' :    {'Reg':3,'IR3':2,'IR2':1,'IR1':0},
                'Utilities':    {'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0},
                'HeatingQC':   {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa':1, 'Po':0},
                'KitchenQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa':1, 'Po':0}           
    }
    
    main.replace(cleanup_dicts, inplace = True)

    #Filling in columns with which values have no ranking, to be converted to integer 0,1,2,...

    column_norank = ['MSZoning','Street','LandSlope','Neighborhood','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
                    'Exterior2nd','Foundation','Heating','CentralAir','Electrical','PavedDrive','SaleType','LandContour','LotConfig',
                    'MasVnrType','Condition1','Condition2','MiscFeature']

    #make subset of dataframe with all columns which have values that cannot be ranked
    df_norank= main.loc[:,column_norank]

    #Change the values for each column to integer 0,1,2,3 .... n, with n is the number of unique value per column
    for column in column_norank:
        count_unique = len(df_norank[column].unique())
        temp_unique = df_norank[column].unique()
        for i in range(count_unique):
            df_norank[column][df_norank[column]==temp_unique[i]] = i

    #Apply above in the main dataframe        
    main.loc[:,column_norank] = df_norank.loc[:,column_norank]


    ## Fill in the NAN with the median values
    main.fillna(main.median(),inplace=True)  

    ##Plotting GrLivArea and SalePrice and find outlier
    #since we see there is one outlier where grLivArea > 4000 but the price is low
    main.drop(main.loc[(main['GrLivArea']>4000) & (main['SalePrice']<300000)].index,inplace=True)

    ##Plotting TotRmsAbvGrd and SalePrice and find outlier
    #since we see there is one outlier where TotRmsAbvGrd > 12 but the price are low
    main.drop(main.loc[(main['TotRmsAbvGrd']>12) & (main['SalePrice']<300000)].index,inplace=True)

    #remove outlier by adjusting the value (typo)
    main.loc[main.GarageYrBlt == 2207, 'GarageYrBlt'] = 2007
    main.loc[main.LotFrontage == 313.0, 'LotFrontage'] = 31.0

    #Remove outlier in LotArea by removing record
    main = main[main.LotArea < 100000]

    #Drop Utilities Column
    main = main.drop('Utilities', 1)

    #Drop Outliers in Electrical column by removing column
    main = main[main.Electrical < 5]

    #Remove records with 1 occurrance
    low_freq = ['Condition2','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterCond','Heating','Electrical',
            'BsmtFullBath','BedroomAbvGr','KitchenAbvGr','KitchenQual', 'TotRmsAbvGrd', 'GarageCars']

    for i in low_freq:
        count = main[i].value_counts()
        main.drop(main[main[i].isin(count[count == 1].index)].index ,inplace=True)

    ##Feature Engineering
    main['TotalSize'] = main['TotalBsmtSF']+ main['GrLivArea']
    main['HouseAge'] = main['YrSold']-main['YearBuilt'] 
    main['Remodeled'] = (main['YearRemod/Add'] != main['YearBuilt']).astype(np.int64)
    main['TotalBathroom'] = main['BsmtFullBath']+main['BsmtHalfBath']/2+main['FullBath']+main['HalfBath']/2 
    main['TtlGarageQ'] = main['GarageFinish'] + main['GarageQual'] + main['GarageCond']

    #Move the SalePrice to the last Column
    temp_SalePrice=main['SalePrice']
    main.drop('SalePrice',axis=1,inplace=True)
    main['SalePrice'] = temp_SalePrice
    
    #After Checking Correlation between all columns and Sale Price
    #Drop Columns which have correlation < 0.01
    main.drop('Foundation',axis=1,inplace=True)
    main.drop('Condition2',axis=1,inplace=True)

    #this is conditional, in general not used, because the data will be normalized in later stage using standard scaler from Scikit Learn
    if normalize==True:    
        ##Normalization with substracting data with the mean and divided by the standard deviation
        temp_saleprice = main['SalePrice']
        main=(main-main.mean())/main.std()    
        main['SalePrice'] = temp_saleprice   #SalePrice to Normal price (without normalization)

    #return(normalized_main)
    return main.reset_index(drop=True)

#### To push through Github