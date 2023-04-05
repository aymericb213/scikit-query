from active_semi_clustering.semi_supervised.pairwise_constraints import COPKMeans
from selection.oracle.MLCLOracle import MaximumQueriesExceeded
from selection import QueryStrategy
import numpy as np



"""
Article :
   Active Image Clustering with Pairwise Constraints from Humans
   Arijit Biswas ·David Jacobs
"""

class Pairwise(QueryStrategy):
   """
     ########
   """   

   def __init__(self ,algo:COPKMeans , nombre_elements : int):
      super().__init__()
      self.algo = algo
      self.nombre_elements = nombre_elements

   def _generer_matrice_probabilite(self,dataset):
      precision = 100
      algo = self.algo
      def generer_matrice(mat,cluster):
         taille_donnee = len(cluster)
         for i in range(taille_donnee):
               for j in range(i+1,taille_donnee):
                  mat[i][j]+= (1.0/precision)  if(cluster[i] == cluster[j]) else 0
                  
      taille_jeu_donnee = len(dataset.data)
      nombre_colonne , nombre_ligne= taille_jeu_donnee , taille_jeu_donnee
      list_cluster = None
      mat = np.array([[0.0]*nombre_colonne]*nombre_ligne) # ou  mat = np.zeros((nb_ligne,nb_colonne))   
      taille_jeu_donnee = len(dataset.data)
      nombre_colonne , nombre_ligne= taille_jeu_donnee , taille_jeu_donnee
      list_cluster = None
      mat = np.array([[0.0]*nombre_colonne]*nombre_ligne) # ou  mat = np.zeros((nb_ligne,nb_colonne))
      
      for iteration in range(precision):
         array = np.copy(algo.fit(dataset.data).labels_)   
         generer_matrice(mat=mat,cluster=array)    
         list_cluster = np.concatenate((list_cluster,array),axis=0) if list_cluster is not None else np.copy(array)
   
      return mat 
   

   

   def fit(self, dataset):
      return self.active_HACC(dataset)
   
   def paire(self,point_ic,point_jc,clusters):
      #print(f" ({point_ic},{point_jc})")
      paire_de_point:bool = clusters[point_ic] == clusters[point_jc]
      return True if paire_de_point else False

   def jcc(self,C1,C2):
      ss , sd , ds = 0,0,0
      for point_i in range(0, self.nombre_elements):
         for point_j in range((point_i + 1),self.nombre_elements):
            if self.paire(point_i,point_j,C1) and self.paire(point_i,point_j,C2):
               ss+=1
            elif self.paire(point_i,point_j,C1) and not(self.paire(point_i,point_j,C2)):
               sd+=1
            elif not(self.paire(point_i,point_j,C1)) and self.paire(point_i,point_j,C2):
               ds+=1
     # print(f"(ss,sd,ds) --> ({ss},{sd},{ds}) ")
      return  0 if ss == 0 else (ss/(ss+sd+ds))


      """
   Ej
   – di and d j are in the same cluster after q questions:
   E J (di ,d j )=|(1 - P (di ,d j ))(1 - J di ,d j ,nq +1 )| 
   – di and d j are in different clusters after q questions:
   E J (di ,d j )=|P (di ,d j )(1 - J di ,d j ,yq +1 )

   (valeur retourné par ej , pair (i,j) , 1 si (i,j) -> dans le  même cluster 0 sinon)
   
   """
   def eJ(self,di,dj,dataset,Mq,Cq): 
      ml = np.copy(Mq) 
      cl = np.copy(Cq) 
      algo = self.algo
      probabilite = 1/algo.n_clusters
      C1 = algo.labels_

      if self.paire(di,dj,C1):

         ml = np.concatenate((ml,(di,dj)),axis=0) if len(Mq) != 0 else np.array([[di,dj]])
         
  
         #print("\nml : " , ml , " \ncl ", cl )
         algo.fit(dataset.data,ml=ml,cl=cl)
         C2 = algo.labels_
         return (abs((1-probabilite)*(1-self.jcc(C1,C2))),(di,dj),0)    
      else:
      
         cl = np.concatenate((cl,[(di,dj)]),axis=0) if len(Cq)!=0 else np.array([[di,dj]])
         print("\nml : " , ml , " \ncl ", cl )
         algo.fit(dataset.data,ml=ml,cl=cl)
         C2 = algo.labels_   
         return (abs((probabilite)*(1-self.jcc(C1,C2))),(di,dj),1)  

   """
   Active-HACC
   D: Jeu de donnée à clusteriser.
   N : Nombre d'élément dans le jeu de donnée .
   K : Nombre de cluster.
   di ,dj : paire d'élément de D.
   Cq : The set of can’t-link constraints obtained after we have
   Mq : The set of must-link constraints obtained after we have
   asked q questions. Note |Cq | + | Mq | = q.
   Hq = HACC(D,Cq ,Mq ): HACC is the clustering function
   on a given dataset D, using the must-link constraints (Mq )
   and can’t-link constraints (Cq ). Hq is the clustering that is
   produced

   """
   def active_HACC(self,D):
      Mq , Cq= np.array([]) , np.array([])
      N = self.nombre_elements 
      algo = self.algo
      constraint = {"Mq":Mq,"Cq":Cq}
      nombre_questions = 0
      paire_interessante = (0,(0,0),0)
      while nombre_questions <= 2:
         for di in range(0,N):
            for dj in range(di+1,N):
               paire_courant = self.eJ(di,dj,D,Mq,Cq) 
               """"
              " if paire_interessante[0] == 0:
                  val_ej : float = paire_courant[0]
                  print(f"\nvaleur de ej := {val_ej}\n")
               """
              # print(" meilleur_paire  := ",paire_interessante,"\npaire_courante",paire_courant,"\n p_c[0] > p_i := ",(paire_courant[0] > paire_interessante[0]))
               if(paire_courant[0] > paire_interessante[0]):
                  paire_interessante = paire_courant    
         
         nombre_questions+=1         
         """
            demande à l'oracle de mettre à jour Mq et Cq 
            en fonctoin de la paire la plus interessante trouvé
         """
          
         if(paire_interessante[0]>0):
           print(f"Ajout \n paire_courant : {paire_courant} , paire_interessante : {paire_interessante}")
           if(paire_interessante[2]):
            Mq = np.concatenate((Mq,[paire_interessante[1]]),axis=0) if len(Mq) != 0 else np.array([paire_interessante[1]])
            print("\nMq : ",Mq)
           else:
            Cq = np.concatenate((Cq,[paire_interessante[1]]),axis=0) if len(Cq)!= 0 else np.array([paire_interessante[1]])
            print("\nCq : ",Cq)
           # print(f"Ajout \n paire_courant : {paire_courant} , paire_interessante : {paire_interessante}")
           constraint = {"Mq":Mq,"Cq":Cq}
      
      #constraint = {"Mq":Mq,"Cq":Cq}
      return constraint    

