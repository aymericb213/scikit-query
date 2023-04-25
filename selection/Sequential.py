import numpy as np
from sklearn.ensemble import RandomForestClassifier
from selection.oracle.MLCLOracle import MaximumQueriesExceeded
from selection import QueryStrategy
from selection.src.BaseSVDD import BaseSVDD
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

"""
Article :
  Active selection of clustering constraints: a sequential approach
  Ahmad Ali Abinn, Hamid Beigy
"""

class Sequential(QueryStrategy):
    

    def __init__(self ,dataset):
        super().__init__()
        self.dataset = dataset
        self.svdds = []


    """
      Version  Optimisée avec numpay
      Déterminer les données à la frontière pour chaque cluster  
    """
    def _svdd_clusters (self):
      
      dataset = self.dataset
      list_clusters = []
      list_svdd = []
     
      """
        Regroupe chaque données d'un même cluster ensemble
      """ 
      for cluster in range(1,int(dataset.n_clusters)+1):
        
        cluster_temporaire = dataset.data[np.where(dataset.labels[0] == cluster)]
        print(len(cluster_temporaire))
        list_clusters.append( cluster_temporaire)
        print(len(list_clusters[cluster-1]))

      """
          Calculer le svdd pour chaque cluster
      """
      for indice in range (int(dataset.n_clusters)):
          X = np.array(list_clusters[indice])
          svdd = BaseSVDD(C=0.9, gamma=0.3, kernel='rbf', display='off')
          # fit the SVDD model
          svdd.fit(X)

          # predict the label
          y_predict = svdd.predict(X)

          list_svdd.append(svdd.boundary_indices)               
            
      print(f"\nfrontière :\n{list_svdd}")   

    """
    Calcul de la distance entre chaque point du dataset avec panda
    """
    def _distance_dataset(self):
        dataset = self.dataset
        distances = pdist(dataset.data)
        dist_matrix = squareform(distances) 
        print(dist_matrix)
        print(f'Min : => {np.min(dist_matrix)}  \nMax : => {np.max(dist_matrix)}') 

    
    """
    Calcul de la distance entre chaque point déterminé par le svdd du dataset avec panda
    """
    def _distance_frontiere():
       pass 
   
    """
    Calcul de la distance entre chaque point d'un clusteur avec panda , 
    effectuer sur l'ensemble des clusters du dataset
    """
    def _distance_clusters():
       pass 
    