# Reproducción de Data Banzhaf

En este trabajo se lleva a cabo una reproducción
del artículo [Data Banzhaf](https://arxiv.org/abs/2205.15466),
siguiendo el estilo del ML Reproducibility
Challenge de [papers with code](https://paperswithcode.com/).

Reproduciremos las secciones $5.1, 5.2$ y $5.3$.

## Sección 5.1
En esta sección se llevan a cabo dos experimentos:

En el primero se compara la eficiencia del MSR
frente al estimador simple de Montecarlo para la estimación del valor de Banzhaf.

En el segundo se comparan los estimadores
obtenidos con dos estimadores populares del valor
de Shapley.

**TODO** list:

- [ ] Generar la muestra a partir de una normal
 bivariante con $\mu = (0.1, -0.1)$ y $\sigma = \mathbb{I}_2$.

 - [ ] No entiendo las gráficas.


## Sección 5.2 
Se busca hacer una comparación de la estabilidad
de los distintos métodos de evaluación ante
perturbaciones de la función de utilidad
producidas por estocasticidad del método
del Descenso del Gradiente Estocástico (SGD).

**TODO** list:

- [ ] Preprocesado de los datos:

    - Fraud, Creditcard, Vehicle y datasets de OpenML:

        - [ ] Subsamplear para balancear las clases en los dataset de OpenML.

        - [ ] Binarizar los dataset multiclase.

    -  CIFAR10 (consultar la bibliografía citada):

        - [ ] Extracción de capas de *ResNet18*.

        - [ ] Extracción de valores de PyTorch.
    
    - MNIST y FMNIST. No se lleva a cabo preprocesado.

- [ ] Ajustar la escala de perturbación provocada
por la estocasticidad del algoritmo de aprendizaje.


## Sección 5.3
En esta sección se llevan a cabo varios experimentos probando la eficacia del valor de Banzhaf en problemas reales de ML.

- **Aprendizaje ponderando las muestras**: Se le asigna un
peso a cada muestra normalizando los valores del dataset entre
$[0,1]$. Durante el entrenamiento cada punto será seleccionado
con probabilidad igual a su peso. Tras esto se entrena un
clasificador y se evalúa sobre el conjunto de test.

- **Detección de muestras mal etiquetadas**: Se estudia la eficacia de distintos métodos de evaluación de datos en la detección de muestras mal etiquedatadas.

(Falta explicar lo de noisy utility function).

**TODO** list:

- [ ] Preprocesado de los datos. (Igual que en 5.2)

- [ ] Seleccionar los puntos a evaluar.
