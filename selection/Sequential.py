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
        self.svdd_clusters = []
        self.svdd_cluster = []
        self.label_svdds = []
        self.distance_svdd = []
        self.max = 0

        self._svdd_clusters()
        self._distance_frontiere()

##////////////////////////////////////////////////////////////////////////////////
    
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
        list_clusters.append( cluster_temporaire)

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
          nombre_frontiere_cluster = len(list_svdd[indice])    
          self.label_svdds.extend([indice]*nombre_frontiere_cluster)         
          self.svdd_clusters = list_svdd
          self.svdd_cluster = np.concatenate(list_svdd).flatten().tolist()
      
      print(f"\nfrontière :\n{list_svdd}\n\nLabel:\n{self.label_svdds}")   

##/////////////////////////////////////////////////////////////////////////////////////////
    
    """
    Calcul de la distance entre chaque point du dataset avec panda
    """
    def _distance_dataset(self):
        dataset = self.dataset
        distances = pdist(dataset.data)
        dist_matrix = squareform(distances) 
        print(dist_matrix)
        print(f'Min : => {np.amin(dist_matrix[dist_matrix !=0])}  \nMax : => {np.max(dist_matrix)}') 

 ##///////////////////////////////////////////////////////////////////////////////////////////////////////   
   
    """
    Calcul de la distance entre chaque point déterminé par le svdd du dataset avec panda
    """
    def _distance_frontiere(self):
        dataset = self.dataset
        list_svdd = self.svdd_clusters
        # Transformer la liste multidimensionnelle en liste simple
        flat_list_svdd_indices =  np.concatenate(list_svdd).flatten().tolist()
        data_svdd_boundary =  dataset.data[flat_list_svdd_indices]
        distances_svdd = pdist(data_svdd_boundary)
        dist_matrix_svdd = squareform(distances_svdd)
        self.distance_svdd = dist_matrix_svdd
        self.max = np.max(dist_matrix_svdd)
 
  ##////////////////////////////////////////////////////////////////////////////////////////////////////////    
    
    """
    Calcul de la distance entre chaque point d'un clusteur avec panda , 
    effectuer sur l'ensemble des clusters du dataset
    """
    def _distance_clusters():
       pass
    
    def fit(self,oracle):
      
      u_t_ij = self.u_indA()
      c_t = []
      ml = []
      cl = []
      
      contraintes = {"ML":ml , "CL":cl}
      
      ####### Hypothèse A ##########
      nombre_question = oracle.budget
      t = 0
       # u_t(c_t,u_t_ij,mat)

      while  t < nombre_question:

          self.u_t(c_t,u_t_ij,cl,ml,False)
          t = t+1

      
      ######### Hypothèse B ###########
      t = 0 
      u_t_ij = self.u_indB()
      while  t < nombre_question:

          self.u_t(c_t,u_t_ij,cl,ml,True)
          t = t+1  

      print(f'\n\n\nImplémentation, Contrainte séquentielle (Ahmad Ali Abinn, Hamid Beigy):\n\n {contraintes}')    
      return contraintes

#////////////////////////////////////////////////////////////////////////////////////////////////
    
    def d_out_ij(self,indice_cluster,indice_donnee1, indice_donnee2):
    
      svdds = self.svdd_clusters
      distance_svdd = self.distance_svdd
      distance = []

      for point_cluster in range(0,len(svdds[indice_cluster])):
          distance.append(distance_svdd[point_cluster,indice_donnee2])
    
      mask_distance = distance[distance != 0]

      indice = np.argmin(mask_distance)

      if(indice_donnee1 > int(indice)):
          distance_retirable  = distance_svdd[indice,indice_donnee1]
      else:
          distance_retirable  = distance_svdd[indice_donnee1,indice]
    
      return distance_retirable

##////////////////////////////////////////////////////////////////////////////////////////////////       
    
    def djc_t( self,c_t , i , j):

        distance_svdd = self.distance_svdd
        # Récupération des 3ème valeurs de chaque sous-tableau
        troisiemes = [x[2] for x in c_t]


        # Récupération de l'indice du minimum des 3ème valeurs
        min_idx = np.argmin(troisiemes)

        # Récupération du minimum des 3ème valeurs
        min_val = troisiemes[min_idx]

        # Récupération des sous-tableaux contenant le minimum
        sous_tableaux = c_t[troisiemes == float(min_val)]
        k = sous_tableaux[0]
        l = sous_tableaux[1]
        dik = distance_svdd[i,k]
        djl = distance_svdd[j,l]
        dil = distance_svdd[i,l]
        djk = distance_svdd[j,k]     
        tableau_min = [(dik + djl) , (dil + djk) ]
        return np.min(tableau_min)

##//////////////////////////////////////////////////////////////////////////////////////
    
    def u_indA(self):

      svdds = self.svdd_cluster 
      label_svdd = self.label_svdds 
      mat_distance = self.distance_svdd
      max = self.max
      taille = len(svdds)

      u_t_ij = np.zeros((taille,taille))


      for i in range(0,taille):
          for j in range((i+1),taille):
              
              ci = label_svdd[i]
              cj = label_svdd[j]
              dij = mat_distance[i,j]
              dijout = dij
              vijout = 0.0
              # Vérifier si i et j ne sont pas dans le même cluster
              if(not(label_svdd[i] == label_svdd[j])):
                  min_ci = self.d_out_ij(ci,i,j)
                  min_cj = self.d_out_ij(cj,j,i)
              
                  dijout-=min_ci
                  dijout-=min_cj

              vijout = dijout / dij
              u_t_ij[i,j] = float(vijout * (1-(dij /max)))    

      return u_t_ij     
        
##////////////////////////////////////////////////////////////////////////////////////
 
    def u_indB(self):

      svdds = self.svdd_cluster 
      label_svdd = self.label_svdds 
      distance_svdd = self.distance_svdd
      max = self.max 
      taille = len(svdds)
      u_t_ij = np.zeros((taille,taille))


      for i in range(0,taille):
          for j in range((i+1),taille):
              
              ci = label_svdd[i]
              cj = label_svdd[j]
              dij = distance_svdd[i,j]
              dijout = dij
              vijout = 0.0
              
              # Vérifier si i et j ne sont pas dans le même cluster
              if(not(label_svdd[i] == label_svdd[j])):
                  min_ci = self.d_out_ij(ci,i,j)
                  min_cj = self.d_out_ij(cj,j,i)
              
                  dijout-=min_ci
                  dijout-=min_cj

              vijout = dijout / dij

              u_t_ij[i,j] = (1-vijout) * (1-(dij/max))    

      return u_t_ij
        
##//////////////////////////////////////////////////////////////////////////////
    
    def u_t(self,c_t, u_t_ij , cl , ml , link):

        svdds = self.svdd_cluster
        max = self.max
        distance_svdd = self.distance_svdd
        taille = len(svdds)

        #indice de la valeur maximale
        max_ij_indice =  np.argmax(u_t_ij)

        #Récupérer les indices x et y correspondants à la matrice
        x,y = np.unravel_index(max_ij_indice,u_t_ij.shape)
        dij = distance_svdd[x,y]
        np_c_t = np.array(c_t)
        
        if(len(c_t) > 0):
            
            #vérifier si le couple (i,j) appartient au tableau
            indices = np.where((np_c_t[:,0]==x)& (np_c_t[:,1] == y))
            indices = np.asarray(indices)
            
            if(indices.size == 0):
                c_t.append([x,y,dij])
                if(link):
                    ml.append([svdds[x],svdds[y]])
                else:
                    cl.append([svdds[x],svdds[y]])    


        else:
            c_t.append([x,y,dij])     
            if(link):
                ml.append([svdds[x],svdds[y]])
            else:
                cl.append([svdds[x],svdds[y]])    

        u_t_ij[x,y] = -1

        for i in range(0,taille):
            for j in range((i+1),taille):
              
                djct = self.djc_t(c_t,i,j)
                u_t_ij[i,j] = (u_t_ij[i,j] * djct) / max





  