# [Foursqure Competition](https://www.kaggle.com/competitions/foursquare-location-matching/overview/description)

The goal of this competition is match POIs together. Using a dataset of over one-and-a-half million Places entries heavily altered to include noise, duplications, extraneous, or incorrect information, it is expected to produce an algorithm that predicts which Place entries represent the same point-of-interest.


# Data

They are three datasets provided: train, test, and pairs. Train and Test data have the same format with train having an extra columns (point_of_interest) where it identifies the point that refers to the entity. For each entity we have the name, url, address, country, zip_code, and etc. The pairs dataset is built upon the training data just matching the pairs and seeing if they are the same or not.


# Feature Generation

Features created are mainly similarity scores generated using the difflib library. In comparision between two entities, same attributes are compared to measure how similar those attributes are. Each pair of attributes will give us one numerical measure of similarity, except for cases when one or both of the attributes are missing (then we put missing in the entry).

```python
def similarity(addr1,addr2):
    if pd.isna(addr1) or pd.isna(addr2): return np.nan
    
    return round(difflib.SequenceMatcher(None, addr1,addr2).ratio(), 4)
```

The function above is used to generate the similarity score:

```python
pairs['name_sim'] = pairs.apply(lambda x: similarity(x.name_1, x.name_2), axis=1)
pairs['addr_sim'] = pairs.apply(lambda x: similarity(x.address_1, x.address_2), axis=1)
pairs['cat_sim'] = pairs.apply(lambda x: similarity(x.categories_1, x.categories_2), axis=1)
pairs['url_sim'] = pairs.apply(lambda x: similarity(x.url_1, x.url_2), axis=1)

pairs['zip_sim'] = pairs.apply(lambda x: similarity(x.zip_1, x.zip_2), axis=1)
pairs['city_sim'] = pairs.apply(lambda x: similarity(x.city_1, x.city_2), axis=1)
pairs['state_sim'] = pairs.apply(lambda x: similarity(x.state_1, x.state_2), axis=1)
```

## Categorical Similarity

The categories column has multiple categories seperated by a comma for each entity.The score of similarity for category columns did not seem to be descriptive hence I wrote a function to calculate how close two entities are. Where the number of categoris they have in common squared is divided the multiplcation of total number of categories for each entity.

```python
def cat_val(cat1, cat2):
    cat1 = str(cat1).replace(' ', '').split(',')
    cat2 = str(cat2).replace(' ', '').split(',')
    
    den = len(cat1) * len(cat2)
    
    nom = len(set(cat1).intersection(set(cat2))) ** 2
    
    return nom / den
```

## Location Differences

Longitutde and Latitudes are given for each point and the difference between these metrics can be helpful.

```python
pairs['lat_diff'] = abs(pairs['latitude_1'] - pairs['latitude_2'])
pairs['lon_diff'] = abs(pairs['longitude_1'] - pairs['longitude_2'])
```

## Adding more data

There are 1 million entites within the training data, and there is only 0.5 million pairs provided. Presumably by generating more pairs (with features), the accuracy of the model will increase.

However there are different ways that this process could be done:
1. Permutations: This is a random process where different ids are matched together. 
2. Balancing the data: The original pairs provided has more matches than non-matches (double), by explicitly generating non-matches from data. By balancing out the matches, model's performance will be more resonable given data it has not seen.
3. Hand-Picking: There might be special cases that we want present in the dataset just for the sake of diversifying the patterns present:
    - Missing values in certain columns, we might want to train models that can make robust predictions given that entries are missing in certain columns.
    - Choosing pairs based on how similar they are in certain attributes (name, url, etc.)
    - Choosing pairs based on the repition of their point_of_interest. Starting with the most repeated point_of_interest, will generate the pairs.

# Modelling

## Experiments

1. Bagging the data using Thresholds

Given multiple threshhold for each column, data is filtered. Meaning that only values that are higher than a threshold is kept. Then a model is trained using this data.

```json
tdic = {
    'url_sim': [0.1, 0.3, 0.61, 0.8, 0.91],
    'cat_union': [0.10, 0.2, 0.33, 0.8],
    'lon_diff': [0.000178, 0.000690, 0.003257],
    'lat_diff': [0.000147, 0.0005, 0.001, 0.002, 10, 50],
    'addr_sim': [0.15, 0.32, 0.55, 0.8, 0.9],
    'zip_sim': [0.05, 0.1, 0.5, 0.95],
    'name_sim': [0.2, 0.41, 0.66, 0.91, 0.95],
    'phone_sim': [0.1, 0.5714, 0.6],
}
```

The end goal is to measure the accuracy of aggregated models trained based on thresholds. For the experiment to be general and to make sure every possible aspect is taken into account they are multiple ways that generated data from models can be used to train more models (ensembling). It is important to note that only lightgbm and Xgboost will be used for training the models. Type of data concatenation possible:

1. Features: Only using the generated features. This would be considered a benchmark where the other datasets are compared to see if their increase in accuracy is significant.
2. Features + xgb predictions
3. Features + lgbm predictions
4. xgb predictions
5. lgbm predictions
6. Features + lgbm predictions + xgb predictions
7. lgbm predictions + xgb predictions 

After this, a few light Neural Network architectures will be trained, so we can also see if a Deep Learning ensemble over the threshold models is feasible or not. Lists below show the layers that each Neural Network will have.

```python
l1 = [tf.keras.layers.Dense(1, activation='sigmoid')]
l2 = [tf.keras.layers.BatchNormalization(), 
      tf.keras.layers.Dense(1, activation='sigmoid')]
l3 = [tf.keras.layers.Dense(8), 
      tf.keras.layers.Dense(1, activation='sigmoid')]
l4 = [tf.keras.layers.Dense(16), 
      tf.keras.layers.Dense(1, activation='sigmoid')]
l5 = [tf.keras.layers.Dense(8), 
      tf.keras.layers.Dense(16), 
      tf.keras.layers.Dense(1, activation='sigmoid')]
```

