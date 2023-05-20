import pandas as pd
import numpy as np
import os
import argparse
import clustbench
from selection import *
from selection.aipc import AIPC
from selection.oracle import MLCLOracle
from active_semi_clustering.semi_supervised.pairwise_constraints import COPKMeans, PCKMeans, MPCKMeans
from sklearn.cluster import *
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import plotly.express as px
import plotly.graph_objects as go

def plot(dataset, partition, constraints=None, filename=None):
    viz_dataset = pd.DataFrame(PCA(n_components=2).fit_transform(dataset)) if dataset.shape[1] > 3 else pd.DataFrame(dataset)
    fig = None
    match viz_dataset.shape[1]:
        case 2:
            fig = px.scatter(viz_dataset, x=0, y=1, template="simple_white",
                            color=partition, symbol=partition,
                             hover_data={'index': viz_dataset.index.astype(str)})
        case 3:
            fig = px.scatter_3d(viz_dataset, x=0, y=1, z=2, template="simple_white",
                            color=partition, symbol=partition,
                            hover_data={'index': viz_dataset.index.astype(str)})
	
    if constraints:
        for key in constraints:
            for cst in constraints[key]:
                points = viz_dataset.iloc[list(cst)]
                match viz_dataset.shape[1]:
                    case 2:
                        fig.add_trace(go.Scatter(name=str(cst), x=[points.iloc[0, 0], points.iloc[1, 0]],
                                                 mode="lines", y=[points.iloc[0, 1], points.iloc[1, 1]]))
                    case 3:
                        fig.add_trace(go.Scatter3d(name=str(cst), x=[points.iloc[0, 0], points.iloc[1, 0]],
                                                   mode="lines",  y=[points.iloc[0, 1], points.iloc[1, 1]],
                                                                  z=[points.iloc[0, 2], points.iloc[1, 2]]))
                if key == "ML":
                    fig['data'][-1]['line']['color'] = "#ff0000"
                else:
                    fig['data'][-1]['line']['color'] = "#0000ff"
                    fig['data'][-1]['line']['dash'] = "dash"

    fig.update_layout(showlegend=False)
    fig.update(layout_coloraxis_showscale=False)
    if not filename:
        fig.show()
    else:
        fig.write_html(filename)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active Clustering')
    parser.add_argument('-path', type=str, help='path to clustbench data folder')
    args = parser.parse_args()

    dataset = clustbench.load_dataset("fcps", "lsun", path=args.path)
    labels = dataset.labels[0] # - 1 if args.auto else None # correspondance between clustbench and Python indexing

    algo = COPKMeans(n_clusters=dataset.n_clusters[0])
    algo.fit(dataset.data)
    init_partition = algo.labels_
    print(adjusted_rand_score(labels, algo.labels_))#ARI entre la partition initiale et la vérité terrain : qualité
    plot(dataset.data, algo.labels_, filename="initial_partition.html")


    """
     active_qs contient la stratégie choisie.
     Il suffit de décommenter la ligne a utiliser
    """

    sequential = Sequential(dataset)
    #constraints = sequential.fit(algo.labels_,MLCLOracle(truth=labels))
    #algo.fit(dataset.data, ml=constraints["ML"], cl=constraints["CL"])
    moyenne = 0
    for i in range(30):
        constraints = sequential.fit(algo.labels_,MLCLOracle(truth=labels,budget=5))
        plot(dataset.data, algo.labels_, constraints, "partition_with_constraints.html")
        print("iteration = ", i)
        print("CL : ", constraints["CL"])
        print("ML : ", constraints["ML"])
        algo.fit(dataset.data, ml=constraints["ML"], cl=constraints["CL"])
        plot(dataset.data, algo.labels_, filename="modified_partition.html")
        print(adjusted_rand_score(init_partition, algo.labels_))#ARI entre la partition de départ et la partition modifiée : similarité
        moyenne += adjusted_rand_score(labels, algo.labels_)#ARI entre la partition modifiée et la vérité terrain : qualité
    moyenne = moyenne / 30
    print(moyenne)
    """
    #active_qs = NPUincr()
    active_pairwise = Pairwise(algo, len(dataset))
    matrice_probabilite = active_pairwise._generer_matrice_probabilite(dataset=dataset)
    print(f'Matrice de probabilité:\n {matrice_probabilite}')
    print(f'échantillon plus claire:\n {matrice_probabilite[0]}')
    #constraints = active_qs.fit(dataset.data, algo.labels_, MLCLOracle(truth=labels, budget=10))
    Sequential = Sequential(dataset)
    constraints = Sequential.fit(MLCLOracle(truth=labels))

    cont = Sequential.fit(algo.labels_,MLCLOracle(truth=labels,budget=100))

    #algo.fit(dataset.data, ml=np.array([ [38, 28], [38, 27]]) , cl= [ [9, 39], [38, 95]])
    print(f" len(ml) => { len(cont['ML'])}  len(cl) => {len(cont['CL'])}")
    algo.fit(dataset.data, ml = cont['ML'] , cl=cont['CL'])
    print(adjusted_rand_score(labels, algo.labels_))
    print(adjusted_rand_score(init_partition, algo.labels_))
    """
