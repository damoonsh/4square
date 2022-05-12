'''
    utilities used for feature generation
'''

def similarity(addr1,addr2):
    if pd.isna(addr1) or pd.isna(addr2): return np.nan
    
    return round(difflib.SequenceMatcher(None, addr1,addr2).ratio(), 4)


def cat_val(cat1, cat2):
    cat1 = str(cat1).replace(' ', '').split(',')
    cat2 = str(cat2).replace(' ', '').split(',')
    
    den = len(cat1) * len(cat2)
    
    nom = len(set(cat1).intersection(set(cat2))) ** 2
    
    return nom / den


def batch_gen(Ids):
    feats = [
        'id_1', 'id_2',
        'lat_diff', 'lon_diff', 'url_sim', 
        'addr_sim', 'name_sim', 'cat_union', 
        'zip_sim', 'city_sim', 'state_sim',
        'phone_sim', 'match'
    ]

    dic = {col:[] for col in feats}

    for i in range(len(Ids)):
        r1 = train[train['id'] == Ids[i]]
        for j in range(i+1,len(Ids)):
            r2 = train[train['id'] == Ids[j]]
            
            dic['match'].append(r1['point_of_interest'].values[0] == r2['point_of_interest'].values[0])
            
            dic['id_1'].append(Ids[i])
            dic['id_2'].append(Ids[j])

            dic['lat_diff'].append(abs(r1['latitude'].values[0] - r2['latitude'].values[0]))
            dic['lon_diff'].append(abs(r1['longitude'].values[0] - r2['longitude'].values[0]))

            dic['cat_union'].append(cat_val(r1['categories'].values[0], r2['categories'].values[0]))

            dic['addr_sim'].append(similarity(r1['address'].values[0], r2['address'].values[0]))
            dic['name_sim'].append(similarity(r1['name'].values[0], r2['name'].values[0]))
            dic['url_sim'].append(similarity(r1['url'].values[0], r2['url'].values[0]))
            dic['city_sim'].append(similarity(r1['city'].values[0], r2['city'].values[0]))
            dic['state_sim'].append(similarity(r1['state'].values[0], r2['state'].values[0]))
            dic['zip_sim'].append(similarity(r1['zip'].values[0], r2['zip'].values[0]))
            dic['phone_sim'].append(similarity(r1['phone'].values[0], r2['phone'].values[0]))

    return pd.DataFrame(dic)