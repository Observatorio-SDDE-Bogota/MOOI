# Se importan las librerías externas a usar
import numpy as np
np.random.seed(0)

import pandas as pd
import fontstyle
import matplotlib.pyplot as plt
from prince import FAMD
import sklearn
from sklearn.experimental import enable_iterative_imputer
from sklearn import linear_model
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
import re
from sklearn.pipeline import Pipeline
import statsmodels.formula.api as smf

from IPython.display import display, HTML


# Se importan los scripts propios a usar
import limpieza_variables as lv
import criterios_programas as cp
import diccionario_variables as dv
import Fun_triage as triage




def triage(nombre_programa, df_tropa, features, features_cont, features_cat, mice_imputer, famd):   
    print(fontstyle.apply('Programa: ' + nombre_programa, 'bold/Italic/black/WHITE_BG'))
    
    np.random.seed(0)

    # %% Dejar solo los registros que cumplen con los criterios del programa utilizando los diccionarios criterios_dict y tipo_dict
    criterios = cp.criterios_dict.get(nombre_programa)
    tipo_exclusion = cp.tipo_dict.get(nombre_programa)
    print(fontstyle.apply('     1. Tipo de criterio: ' + tipo_exclusion, 'black'))

    if tipo_exclusion == 'Excluyente':
        df_tropa_programa = df_tropa.loc[(df_tropa[list(criterios)] == pd.Series(criterios)).all(axis=1)]
    else:
        df_tropa_programa = df_tropa.loc[(df_tropa[list(criterios)] == pd.Series(criterios)).any(axis=1)]


    # %% Elección de variables - ver en criterios_programas.py



    # %% Imputación de datos
    #print(fontstyle.apply('     2. Estadísticas descriptivas antes de imputación', 'bold/Italic/black'))
    x = df_tropa_programa.loc[:, cp.features]
    #display(x.describe())
    x_imp = x.loc[:, cp.features_cont]
    x_imp = x_imp.values
    x_imp = mice_imputer.fit_transform(x_imp)
    #imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') # si se quiere imputar solo con la media
    #x_imp = imp_mean.fit_transform(x_imp)
    x_imp = pd.DataFrame(x_imp, columns=cp.features_cont)
    for col in cp.features_cont:
        x[col] = x_imp[col].values
        
    print(fontstyle.apply('     2. Estadísticas descriptivas después de imputación MICE', 'black'))
    display(x.describe())
    # display(x['INGRESOS'].sum()) ## borrar


    # %% Se se ajusta a los datos utilizando FAMD
    print(fontstyle.apply('     3. Se reduce dimensionalidad utilizando análisis de factores para datos mixtos', 'black'))
    df_famd = famd.fit(x)
    
    
    #Esta función mapea como se relacionan los componentes y la variable de elección, en este caso se elige ANTIGUEDAD
    #x['ANTIGUEDAD'] = df_tropa['ANTIGUEDAD']
    #famd_img = famd.plot_row_coordinates(x, ax=None, figsize=(6, 6),
    #                                      x_component=0, y_component=1,
    #                                      labels=None,
    #                                      color_labels=['antigüedad {}'.format(t) for t in x['ANTIGUEDAD']],
    #                                      ellipse_outline=True,
    #                                      ellipse_fill=False,
    #                                      show_points=True)
        
    
    
    # %%Cálculo del número de clústers ideal para k-means
    print(fontstyle.apply('     4. Definición número de clusters utilizando el método Elbow', 'black'))


    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 10)
    rc = df_famd.row_coordinates(x)
 
    
    for k in K:
    # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k,
                            random_state=42)
        kmeanModel.fit(rc)
    
        distortions.append(sum(np.min(cdist(rc, kmeanModel.cluster_centers_,
                                             'euclidean'), axis=1)) / rc.shape[0])
        inertias.append(kmeanModel.inertia_)
    
        mapping1[k] = sum(np.min(cdist(rc, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / rc.shape[0]
        mapping2[k] = kmeanModel.inertia_
        
    
    print('Distorsión')
    for key, val in mapping1.items():
         print(f'{key} : {val}')
    #print('Inercia')
    #for key, val in mapping2.items():
    #    print(f'{key} : {val}')
    
     
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Número de clusters (K)')
    plt.ylabel('Distorsión')
    plt.title('Método Elbow usando distorsión')
    plt.show()

    #plt.plot(K, inertias, 'bx-')
    #plt.xlabel('Número de clusters (K)')
    #plt.ylabel('Inercia')
    #plt.title('Método Elbow usando inercia')
    #plt.show()    
    
    
    
 
    # %% Se inicializa el modelo de k-means
    print(fontstyle.apply('     5. Clusterización utilizando KMeans', 'black'))

    num_clusters = 3 # se define a partir de los resultados del metodo Elbow
    km = KMeans(n_clusters=num_clusters, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04, 
            random_state=42)
    

    
    y_km = km.fit_predict(df_famd.row_coordinates(x))
    
    x_dist = km.transform(df_famd.row_coordinates(x)) ** 2
    
#    %% Gráfica de k-means
#    plot the 3 clusters
    fx = pd.DataFrame(df_famd.row_coordinates(x), columns=[0, 1])
    fx['kmeans'] = y_km

    
    plt.scatter(
         fx[fx['kmeans'] == 0][0], fx[fx['kmeans'] == 0][1],
         s=25, c='lightgreen',
         marker='o', edgecolor=None,
         label='cluster 1'
     )
    
    plt.scatter(
         fx[fx['kmeans'] == 1][0], fx[fx['kmeans'] == 1][1],
         s=25, c='orange',
         marker='o', edgecolor=None,
         label='cluster 2'
     )
    
    plt.scatter(
         fx[fx['kmeans'] == 2][0], fx[fx['kmeans'] == 2][1],
         s=25, c='lightblue',
         marker='o', edgecolor=None,
         label='cluster 3'
     )
    
    #plt.scatter(
    #     fx[fx['kmeans'] == 3][0], fx[fx['kmeans'] == 3][1],
    #     s=25, c='purple',
    #     marker='o', edgecolor=None,
    #     label='cluster 4'
    # )
    
    # plot the centroids
    plt.scatter(
         km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
         s=80, marker='o',
         c='red', edgecolor='black',
         label='centroids'
     )
    plt.legend(scatterpoints=1)
    plt.title('Clusters de unidades productivas utilizando KMeans')
    plt.grid()
    plt.show()    

    
   # %% Descripción de resultados
    print(fontstyle.apply('     6. Caracterísiticas de cada cluster', 'black'))
    # display(x['INGRESOS'].sum()) ## borrar
    x_res = x
    x_res['kmeans'] = fx['kmeans']    
    x_res = x_res.reset_index()
    x_dist = pd.DataFrame(x_dist)
    x_dist['min_dist'] = x_dist[[0,1,2]].min(axis=1)
    x_dist['min_dist'] = 100 - (((x_dist['min_dist'] - x_dist['min_dist'].min()) / (x_dist['min_dist'].max() - x_dist['min_dist'].min())) * 100)
    
    x_res = pd.concat([x_res, x_dist], axis=1, join="inner")
    x_res[features_cat] = x_res[features_cat].astype(str)
    x_catd = pd.get_dummies(x_res[features_cat])
    x_res = pd.concat([x_res, x_catd], axis=1, join="inner")
    
    # display(x_res['INGRESOS'].sum()) ## borrar
    display(x_res.groupby('kmeans',sort=False).count())

    resultados = x_res.drop(['DIGITAL', 'ANTIGUEDAD', 'R_MERCANTIL', 'RAZON_CREAR_NEGOCIO'], axis=1).groupby(['kmeans']).agg(['mean', 'median', 'std']).transpose()

    x_res = x
    x_res['kmeans'] = fx['kmeans']
    x_res['x'] = fx[0]
    x_res['y'] = fx[1]
    x_res = x_res.reset_index()

    cluster_center = km.cluster_centers_[:, 0], km.cluster_centers_[:, 1]
    cluster_center = pd.DataFrame(cluster_center).transpose().reset_index()
    cluster_center = cluster_center.rename(columns = {'index':'kmeans', 0:'x_mean', 1:'y_mean'})

    x_res = x_res.merge(cluster_center, left_on='kmeans', right_on='kmeans')
    x_res['dist'] = np.sqrt((x_res['x'] - x_res['x_mean'])**2 + (x_res['y'] - x_res['y_mean'])**2)

    
    #Seleccionar top N
    x_res[features_cat] = x_res[features_cat].astype(str)
    x_catd = pd.get_dummies(x_res[features_cat])
    x_res = pd.concat([x_res, x_catd], axis=1, join="inner")
    resultados = x_res.drop(['DIGITAL', 'ANTIGUEDAD', 'R_MERCANTIL', 'RAZON_CREAR_NEGOCIO'], axis=1).groupby(['kmeans']).agg(['mean', 'median', 'std']).transpose()
    #print(x_beneficiarios['kmeans'].value_counts())
    display(resultados)

    # Numerical variables
    num_variables = ['EDAD', 'INGRESOS', 'NO_TRABAJADORES']
    grouped_df = x_res.groupby('kmeans')


    resultados = resultados.dropna(axis=1)
    seleccion_cluster = resultados[resultados.index == ('INGRESOS',   'mean')].min().to_frame().idxmin()
    seleccion_cluster['INGRESOS/mean'] = resultados[resultados.index == ('INGRESOS',   'mean')].min().idxmin()
    seleccion_cluster['INGRESOS/std'] = resultados[resultados.index == ('INGRESOS',   'std')].min().idxmin()
    seleccion_cluster['EDAD/mean'] = resultados[resultados.index == ('EDAD',   'mean')].min().idxmin()
    seleccion_cluster['NO_TRABAJADORES/mean'] = resultados[resultados.index == ('NO_TRABAJADORES',   'mean')].min().idxmin()
    seleccion_cluster['PROPIETARIOS_HOMBRES_dummy/mean'] = resultados[resultados.index == ('PROPIETARIOS_HOMBRES_dummy', 'mean')].max().idxmax()
    seleccion_cluster = seleccion_cluster.to_frame()
    cluster = seleccion_cluster[0].value_counts().idxmax()

    x_mean_chosen = cluster_center.loc[cluster_center['kmeans'] == cluster, 'x_mean'].values[0]
    y_mean_chosen = cluster_center.loc[cluster_center['kmeans'] == cluster, 'y_mean'].values[0]

    print(fontstyle.apply('     7. Puntaje de cada unidad productiva según su distancia al centroide del cluster seleccionado', 'black'))

    x_res['x_mean_chosen'] = x_mean_chosen
    x_res['y_mean_chosen'] = y_mean_chosen
    x_res['dist_chosen'] = np.sqrt((x_res['x'] - x_res['x_mean_chosen'])**2 + (x_res['y'] - x_res['y_mean_chosen'])**2)
    x_res['min_dist_order'] = 100 - (((x_res['dist_chosen'] - x_res['dist_chosen'].min()) / (x_res['dist_chosen'].max() - x_res['dist_chosen'].min())) * 100)

    x_res[nombre_programa] = x_res['min_dist_order']
    x_res = x_res.set_index(x_res.iloc[:, 0].name)
    display(x_res)
    return x_res