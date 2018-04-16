import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#get 49906+5469 data
data=fetch_movielens(min_rating=4.0)

#show how many training and test data in the dataset
print(repr(data['train']))
print(repr(data['test']))

#just want to see how "shape" works and how dataset looks like
print(data['train'].shape)
print(data['test'])

#define model parameters
model=LightFM(loss='warp')

#feed training data to model
model.fit(data['train'], epochs=30, num_threads=2)

#define function for recommandation
def sample_recommendation(model, data, user_ids):
#use "n_users" , "n_items" to record the dimension of dataset
  n_users, n_items=data['train'].shape
  print(n_users)
  print(n_items)
#iterate through "user_ids" array
  for user_id in user_ids:
    known_positives=data['item_labels'][data['train'].tocsr()[user_id].indices]
    scores=model.predict(user_id, np.arange(n_items))
    top_items=data['item_labels'][np.argsort(-scores)]
    print("User %s" % user_id)
    print("Known positives:")

    for x in known_positives[:3]:
      print(" %s" % x)

    print("Recommended:")

    for x in top_items[:3]:
      print(" %s" % x)

sample_recommendation(model, data, [3,25,450])