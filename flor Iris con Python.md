## Flor Iris con Python. ( Creando un clasificador para la flor Iris con Python)



Este proyecto es uno de los básicos,  que se utilizan al momento de estar aprendiendo Machine Learning.



La idea principal es construir un modelo que pueda predecir a qué tipo de flor pertenece una muestra, basándose en características como el tamaño de sus pétalos y sépalos, para ello vamos a construir un clasificador para el Dataset de  las flores tipo Iris. 

En resumen , el dataset Iris es un problema de clasificación supervisada, donde usamos los 4 atributos (features/entradas) para predecir la clase de flor (Species/salida)

### El dataset de Iris: 

El conjunto de datos contiene 50 muestras de cada una de tres especies de Iris (Iris setosa, Iris virginica e Iris versicolor). 

Se midió cuatro rasgos de cada muestra: el largo y ancho del sépalo y pétalo, en centímetros:
```
1 - Características de entrada (features)
      
            Largo del sépalo (sepal length) en cm.

            Ancho del sépalo (sepal width) en cm. 

            Largo del pétalo (petal length) en cm.

            Ancho del pétalo (petal width) en cm.

2 - Etiqueta o clase (target):

                Iris setosa

                Iris versicolor

                Iris virginica
```

![image](https://github.com/user-attachments/assets/dda17548-1d7e-4826-8a1d-0a1e236694aa)
![image](https://github.com/user-attachments/assets/edd3a51a-4ac5-45a0-b9d8-7bd37e9a66d3)
![image](https://github.com/user-attachments/assets/b697eab7-2dbf-4e61-8ce5-47bfa936575a)



## Codigo python


### 1 - Importacion de Librerias

```py
# importamos librerias para nuestro proyecto.
# librerias numpy y pandas: Manejo y análisis de datos numéricos y estructurados.

import numpy as np
import pandas as pd

# librerias para creacion y  visualización gráfica de datos.

import matplotlib.pyplot as plt
import seaborn as sns
```

```py
# Libreria sklearn: Implementa algoritmos de Machine Learning y evaluación
import sklearn

from sklearn.model_selection import train_test_split         # Divide un conjunto de datos en dos partes: datosentrenamiento y datos de prueba.

from sklearn.ensemble import RandomForestClassifier          # Implementa un modelo de bosque aleatorio.
from sklearn.tree import DecisionTreeClassifier              # Construye un árbol de decisiones.
from sklearn.neighbors import KNeighborsClassifier           # Clasifica un dato basándose en los vecinos más cercanos.
from sklearn.linear_model import LogisticRegression          # modelo de clasificación 
from sklearn.svm import SVC                                  # Implementa Máquinas de Vectores de Soporte (Support Vector Machines). 

# Importar las métricas necesarias
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
```


### 2 - Carga y Lectura de Datos
```py
# leemos el archivo CSV, llamado Iris.csv y lo cargamos o en un DataFrame de pandas llamado iris.

iris = pd.read_csv("Iris.csv")
```


### 3 - Entender los datos del Dataset

```py
# Devuelve una tupla (n_filas, n_columnas): Muestra el tamaño del DataFrame
# salida: (150, 6)

iris.shape
```




```py
# Obtener el número total de elementos del dataframe
# es el producto del número de filas y columnas ==> 150x6 = 900

iris.size 
```


```py
# El método .head() en pandas devuelve las primeras filas de un DataFrame, lo que te permite echar un vistazo rápido al contenido de los datos.
# Por defecto, muestra las primeras 5 filas

iris.head()
```

Salida:

![image](https://github.com/user-attachments/assets/058d5712-eb55-4628-995c-b472a13403ab)




```py
# Resumen general de la estructura del dataset (nombres de columnas, tipos de datos, valores nulos, etc.).

iris.info()
```

![image](https://github.com/user-attachments/assets/338eb873-b269-436c-8aed-daf231616596)

```py
Como podemos observar, todas las columnas contienen 150 datos, en las primeras tenemos datos flotantes mientras que la última contiene datos objetos y es justamente acá en donde se encuentra la información de las especies de la flor.
```
```py
# verificamos la distribución de los datos de acuerdo a las especies de Iris.
# Cuenta la cantidad de registros por especie (Species).

print('Distribución de las especies de Iris:')
print(iris.groupby('Species').size())

```
![image](https://github.com/user-attachments/assets/db617603-5e74-4ba6-9ee1-2f15e150867f)

```txt
Como podemos observar tenemos 50 datos/muestras para cada una de las especies() Iris setosa, Iris versicolor e Iris Virginica).
```

```py
#Revisar si hay valores nulos
print("\nValores nulos por columna:")
print(iris.isnull().sum())
```
![image](https://github.com/user-attachments/assets/92f10e6e-b289-4605-bc6d-eb8eeb259188)

```py
# Estadísticas básicas de las columnas numéricas (media, desviación estándar, etc.).

iris.describe()
```
![image](https://github.com/user-attachments/assets/e877ffed-e837-4555-8ae3-df4682d85859)


### 4 - Transforaciones
```py
# Eliminar la columna 'Id' que no es irrelevante para el análisis , se elimina para evitar que afecte los resultados.
iris = iris.drop(columns=['Id'])

iris.head()
```

![image](https://github.com/user-attachments/assets/0003edb2-281d-4967-b54c-a27be53dd063)


### 5 - Visualización de los datos

Ahora, vamos a representar gráficamente la información para que sea más fácil de entender y analizar. 
Asi, de esta forma, nos pueda permitir  identificar patrones, tendencias y anomalías en los datos de manera más intuitiva que simplemente observando tablas o números.

```py
# Gráficos de barras: Comparan categorías.
# Vemos las cantidades de cada tipo de flor


# Configurar el tamaño de la figura
plt.figure(figsize=(8, 6))

# Gráfico de conteo de especies sin advertencia
sns.countplot(x="Species", data=iris, hue="Species", palette="viridis", legend=False)

# Etiquetas y título
plt.title("Distribución de Especies en el Dataset Iris")
plt.xlabel("Especie de Flor")
plt.ylabel("Cantidad")

# Mostrar el gráfico
plt.show()
```
![image](https://github.com/user-attachments/assets/bc534685-eec8-4fc2-a8e6-4fc70ae68b02)


```py
# Diagramas de dispersión: Analizan relaciones entre dos variables

Comparamos el ancho del sépalo y del pétalo de cada flor, para comprobar si existe relación.

El siguiente código crea dos gráficos de dispersión apilados:

    1️⃣ El primero muestra la relación entre el ancho (PetalWidthCm) y largo del pétalo (PetalLengthCm) por especie.

    2️⃣ El segundo muestra la relación entre el ancho (SepalWidthCm) y largo del sépalo (SepalLengthCm) por especie.  
```



```py
# Crear la figura y los ejes
f, ax = plt.subplots(2,  # Número de filas
                     sharex=False,  # No compartir eje X
                     gridspec_kw={"height_ratios": (1, 1)},  # Proporción igual para los gráficos
                     figsize=(10, 15))  # Tamaño de la figura

# Primer gráfico: Relación entre ancho y largo del pétalo
sns.scatterplot(x=iris['PetalWidthCm'],
                y=iris['PetalLengthCm'],
                hue=iris['Species'],
                palette="viridis",
                ax=ax[0])

# Mejoras del primer gráfico
ax[0].set_title("Relación entre el ancho y largo del pétalo", fontsize=14)
ax[0].set_xlabel("Ancho del pétalo (PetalWidthCm)", fontsize=12)
ax[0].set_ylabel("Largo del pétalo (PetalLengthCm)", fontsize=12)
ax[0].legend(title="Especie", loc="upper left")

# Segundo gráfico: Relación entre ancho y largo del sépalo
sns.scatterplot(x=iris['SepalWidthCm'],
                y=iris['SepalLengthCm'],
                hue=iris['Species'],
                palette="viridis",
                ax=ax[1])

# Mejoras del segundo gráfico
ax[1].set_title("Relación entre el ancho y largo del sépalo", fontsize=14)
ax[1].set_xlabel("Ancho del sépalo (SepalWidthCm)", fontsize=12)
ax[1].set_ylabel("Largo del sépalo (SepalLengthCm)", fontsize=12)
ax[1].legend(title="Especie", loc="upper left")

# Ajustar el espacio entre gráficos
plt.tight_layout()

# Mostrar el gráfico
plt.show(
```



![image](https://github.com/user-attachments/assets/440c5b82-5af9-4094-a49a-a6c7fd260278)


![image](https://github.com/user-attachments/assets/2cccebc2-3a40-4b71-962a-6efca3d6c4b7)


Mirando los graficos, podemos sacar conclusiones:


 ¿Las especies tienen agrupaciones bien definidas o están mezcladas?

 Esto puede sugerir cómo de diferenciadas son las especies en función de estas características.

 Si ves que los puntos de una especie están más juntos o forman una línea o una forma específica, podría indicar que esas especies tienen una relación más fuerte entre el largo y el ancho del pétalo.


 Si las especies están claramente separadas, podrías concluir que el largo y el ancho del pétalo son características muy útiles para distinguirlas. Si las especies están mezcladas, entonces esta relación no es tan diferenciadora. ( verifica si las especies están bien separadas o si hay superposición.)


Si alguna de las especies tiene un solapamiento significativo en las mediciones del sépalo, esto podría sugerir que el sépalo no es tan útil para distinguir esas especies en particular.

¿Hay alguna especie que tenga una tendencia más clara en la forma de dispersión (por ejemplo, si una especie tiene sépalos más grandes o más pequeños en comparación con las otras)?




 Si ambos gráficos muestran que las especies están claramente diferenciadas por las dimensiones del pétalo y el sépalo (tanto en ancho como en largo), entonces esas características son útiles para la clasificación. Puedes decir que el largo y el ancho del pétalo son particularmente buenos para diferenciar las especies.

 Si observas que alguna especie tiene mucho solapamiento con otras, podría indicar que esas especies no son tan fácilmente diferenciables solo con esas dos características, y quizás debas considerar otras medidas (como el ancho y largo de los pétalos y sépalos en combinación con otras características).

 Si los gráficos muestran una relación fuerte entre las dimensiones del pétalo o del sépalo (por ejemplo, largo y ancho del pétalo varían en conjunto), esto podría indicar que hay una correlación en esas dimensiones que puede ser aprovechada en modelos predictivos o análisis futuros.

**En resumen**, la clave está en observar cómo se agrupan o separan las especies en estos dos gráficos. Si las especies se separan bien, puedes confirmar que las características del pétalo y el sépalo son útiles para diferenciar las especies.


```py
# Graficos de Densidad: se usa para visualizar la distribución de una variable numérica

sns.FacetGrid(iris, hue="Species", height=6).map(sns.kdeplot, "PetalLengthCm").add_legend()

sns.FacetGrid(iris, hue="Species", height=6).map(sns.kdeplot, "PetalWidthCm").add_legend()

sns.FacetGrid(iris, hue="Species", height=6).map(sns.kdeplot, "SepalLengthCm").add_legend()

sns.FacetGrid(iris, hue="Species", height=6).map(sns.kdeplot, "SepalWidthCm").add_legend()
```

![image](https://github.com/user-attachments/assets/01fa109f-f14d-4eaf-ac9c-e3daf54567bf)



![image](https://github.com/user-attachments/assets/673da407-9c82-4ba8-8904-fcc74f553c29)



💡 Conclusión 1 : sugiere que la longitud del pétalo(PetalLengthCm) es una variable clave para la clasificación/diferenciar de especies.

                  Setosa tiene pétalos más cortos y concentrados en un rango pequeño.

                  Versicolor y Virginica tienen pétalos más largos, pero con cierta superposición


💡 Conclusión 2:  Sin embargo, hay menos solapamiento que en PetalLengthCm, lo que sugiere que el ancho del pétalo (PetalWidthCm) también es un buen diferenciador de especies.

                  Setosa tiene valores mucho más bajos, mientras que Versicolor y Virginica se superponen en algunos puntos.

### 6 - Modelos de Machine Learning.

este es un ejercicio de clasificación, por lo que todos los algoritmos a implementar deberán ser para este tipo de problemas

Comencemos a construir nuestros modelos con todos los datos, lo primero que debemos hacer es separar los datos con las características, que vendrían siendo todas las columnas (los vamos a llamar “X”), menos la columna de especies (la que llamamos variable “y”).

```py
# Separar las variables independientes y dependientes


       
# se elimina la columna 'Species' del DataFrame iris, o sea, se crea un nuevo DataFrame sin la columna Species y se asigna a la variable X.
# Ahora, X contiene solo las variables independientes.
X = iris.drop(columns=['Species']) 
                       
# Extrae la columna Species del DataFrame iris y la asigna a la variable "y".
# Ahora, y contiene los valores de la variable dependiente, es decir, las especies de las flores.
y = iris['Species']
```


¿Por qué es necesario?

La separación entre variables independientes (X) y la variable dependiente (y) es un paso estándar en Machine Learning. La mayoría de los modelos requieren esta separación porque:

X (características) es lo que el modelo usa para aprender patrones. y (objetivo) es lo que el modelo intenta predecir.

Sin esta separación, no podrías entrenar un modelo correctamente,ya que no sabrá qué datos usar como entrada (X) ni qué valores predecir (y).


Ahora, para entrenar nuestro modelo de Machine Learning y poder saber si esta funcionando correctamente, dividimos nuestro dataset de datos en conjuntos de entrenamiento y prueba.


```py
# X: Las variables independientes (características
# y: La variable dependiente (objetivo).
# test_size=0.2 --> Proporción del conjunto de prueba : 20% de los datos para prueba y el 80% restante para entrenamiento.
# random_state=42 :  Controla la aleatoriedad en la división de datos.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))
```

Salida:

![image](https://github.com/user-attachments/assets/801e2b00-0fcf-48a8-acb8-e190d13ca94a)


El método train_test_split devuelve 4 conjuntos:

Conjunto de entrenamiento: Usado para entrenar el modelo.

        Variables independientes: X_train

        Variable dependiente: y_train

Conjunto de prueba: Usado para evaluar el modelo una vez entrenado.

        Variables independientes: X_test

        Variable dependiente: y_test





¿Por qué es necesario dividir los datos?

La división en conjuntos de entrenamiento y prueba es crucial para construir un modelo de Machine Learning confiable:

Entrenamiento (X_train, y_train): El modelo aprende patrones en los datos. 

Prueba (X_test, y_test): Permite evaluar el rendimiento del modelo en datos no vistos (que simulan nuevos datos en el mundo real).

¿Qué pasa si no lo haces?

 Si no divides los datos y usas todo para entrenar:

El modelo puede memorizar los datos en lugar de generalizar patrones. Esto se llama sobreajuste (overfitting). No podrías evaluar si el modelo funcionará bien con nuevos datos.



Propósito: Verificar qué tan bien el modelo aprendió los datos que se le proporcionaron durante el entrenamiento.

Aquí el modelo tiene ventaja porque ya vio estos datos y puede "memorizar" patrones específicos.

Conjunto de Prueba (X_test, y_test):

Propósito: Medir qué tan bien el modelo generaliza a datos que nunca ha visto antes. Es más importante porque refleja el rendimiento del modelo en datos del mundo real

#### 6.1 - Ahora si empecemos a aplicar los algoritmos de Machine Learning

##### Como forma de aprendizaje , en primer lugar usaremos el modelo de Regresion Logistica

```py
# Creación del modelo de Regresión Logística.
# El método .fit(X_train, y_train) se usa para entrenar el modelo.
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_train, y_train)                                   
```
Ahora vamos a hacer predicciones con el modelo ya entrenado,donde tomamos los datos X_test (que nunca fueron vistos por el modelo durante el entrenamiento).
usa el modelo entrenado para predecir nuevas etiquetas.

```py
# Devuelve una lista y_pred con las clases predichas (las categorías a las que el modelo cree que pertenecen los datos).
y_pred = modelo.predict(X_test)
```
Las predicciones y_pred pueden compararse con los valores reales y_test para evaluar el modelo.


```py
# Calcular las métricas de desempeño
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

# Crear y mostrar la tabla con las métricas
metrics = pd.DataFrame({
    "Métrica": ["Exactitud", "Precisión", "Recall", "F1 Score"],
    "Valor": [accuracy, precision, recall, f1]
})

# Mostrar las métricas en una tabla visual
fig, ax = plt.subplots(figsize=(5, 2))
ax.axis('tight')
ax.axis('off')
table_metrics = ax.table(cellText=metrics.values, colLabels=metrics.columns, loc='center')

plt.show()

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión con un mapa de calor
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))

# Resaltar los labels de los ejes
plt.xlabel("Predicción", fontsize=12, fontweight="bold", color="blue")  # Verde
plt.ylabel("Real", fontsize=12, fontweight="bold", color="green")  # Amarillo

plt.title("Matriz de Confusión")
plt.show()

```
![image](https://github.com/user-attachments/assets/c78d6224-7c5b-42d1-bfd7-aa5d8221205f)




el modelo predijo correctamente todas las muestras en el conjunto de prueba, o sea, el modelo es capaz de generalizar correctamente.
```py
NOTA:El dataset Iris es un conjunto de datos clásico, bien balanceado y pequeño, con solo 150 muestras (50 muestras por cada una de las 3 especies).

Sin embargo, sería recomendable validar el modelo con otros conjuntos de datos o realizar una validación cruzada para asegurarte de que el modelo no esté sobreajustado a los datos.

```
### Ahora,a modo de ejemplo, cambiamos los valarores de los datos para entrenamiento y prueba: test_size=0.6  (60 datos para entrenamiento y 90 datos para prueba) y observamos que pasa.
```py
# Son 60 datos para entrenamiento y 90 datos para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
```
![image](https://github.com/user-attachments/assets/06aaf587-7b58-4cd0-bbfc-22e1eb7dac4d)


![image](https://github.com/user-attachments/assets/cf791fb5-d1ef-47b5-8258-ac282a11dd79)



