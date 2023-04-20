import numpy as np
from sklearn.ensemble import RandomForestClassifier
from selection.oracle.MLCLOracle import MaximumQueriesExceeded
from selection import QueryStrategy
from selection.src.BaseSVDD import BaseSVDD

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
      Déterminer les données à la frontière pour chaque cluster  
    """
    def _svdd_clusters (self):
      
      dataset = self.dataset
      list_clusters = []
      list_svdd = []
     
      """
        Initialiser une liste qui contiendra des clusters et une autre qui aura les frontières de ces derniers 
        en fonction de cluster
      """
      for  indice in range ( int(dataset.n_clusters)):
        list_clusters.append([])
        list_svdd.append([])
      
      """
        Regroupe chaque données d'un même cluster ensemble
      """
      for  indice in range ( len(dataset.data)):
        numero_cluster = dataset.labels[0][indice]
        list_clusters[numero_cluster-1].append(dataset.data[indice]) 
  
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

        list_svdd[indice].append(svdd.boundary_indices)               
          
      print(f"\nfrontière :\n{list_svdd}")
      self.svdds = list_svdd


    """
    Calcul de la distance entre chaque point du dataset avec panda
    """
    def _distance_dataset():
       pass
    
    """
    Calcul de la distance entre chaque point d'un clusteur avec panda , 
    effectuer sur l'ensemble des clusters du dataset
    """
    def _distance_clusters():
       pass 
    