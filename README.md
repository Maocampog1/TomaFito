# TomaFito
# Clasificación de Enfermedades en Hojas de Tomate mediante Redes Neuronales Convolucionales

---

## 1. Problema a Resolver

El tomate (*Solanum lycopersicum*) es uno de los cultivos hortícolas más importantes de Colombia. Según datos de Bayer Agro Colombia, el país siembra aproximadamente **9.000 hectáreas** de tomate y produce alrededor de **512.000 toneladas al año**, con un rendimiento promedio de 62,3 toneladas por hectárea. El cultivo está presente en al menos 21 departamentos, siendo Boyacá, Caldas, Risaralda y Cundinamarca los principales productores. Las variedades más cultivadas son el Milano y el Chonto, ampliamente consumidas en la cocina colombiana.

A pesar de su relevancia económica, el cultivo de tomate es altamente vulnerable a enfermedades foliares. Cuando una hoja de tomate se ve afectada por patógenos como hongos, bacterias o virus, las consecuencias no se limitan a la hoja en sí: la planta pierde capacidad fotosintética, lo que reduce directamente el tamaño, la calidad y el número de frutos producidos. Enfermedades como el tizón tardío (*Late Blight*) —la misma que devastó los cultivos de papa en Irlanda en el siglo XIX— pueden arrasar un cultivo completo en cuestión de días si no se detectan a tiempo. Otras como el virus del enrollamiento amarillo (*Tomato Yellow Leaf Curl Virus*) no tienen cura una vez establecidas, por lo que la detección temprana es la única estrategia efectiva.

El problema central es que la mayoría de los agricultores colombianos, especialmente los pequeños productores, no tienen acceso inmediato a un agrónomo experto. El diagnóstico visual de enfermedades requiere formación especializada y tiempo, y cuando finalmente se identifica el problema, la enfermedad ya puede haber avanzado significativamente. Esto se traduce en pérdidas económicas directas para el agricultor y en desperdicio de alimentos que afectan el abastecimiento regional.

**¿Qué proponemos hacer?** Desarrollar un modelo de clasificación de enfermedades en hojas de tomate basado en redes neuronales convolucionales con transfer learning. El agricultor simplemente toma una foto de la hoja afectada y el modelo identifica automáticamente cuál de las 10 enfermedades más comunes está presente, o confirma que la planta está sana. El objetivo es un modelo liviano que pueda funcionar sin conexión a internet, directamente desde un dispositivo móvil, permitiendo una respuesta inmediata en campo.

---

## 2. Datos

**Dataset utilizado:** Tomato Leaves Dataset — Kaggle  
**Enlace:** https://www.kaggle.com/datasets/ashishmotwani/tomato  
**Tamaño:** 1.47 GB · 32.500 archivos de imagen

El dataset fue recopilado de múltiples fuentes por Ashish Motwani y Khan (2022), combinando imágenes provenientes de dos entornos distintos: escenas de laboratorio con fondo controlado (derivadas del dataset PlantVillage original) y escenas en campo abierto (*in-the-wild*), lo que le otorga mayor capacidad de generalización a condiciones reales de cultivo.

Las imágenes están organizadas en carpetas por clase, listas para ser cargadas directamente con `ImageFolder` de PyTorch. El dataset ya viene pre-dividido en dos splits:

| Split | Descripción |
|-------|------------|
| `train/` | Imágenes para entrenamiento |
| `valid/` | Imágenes para validación |

Para este proyecto aplicaremos la siguiente partición:

| Conjunto | Porcentaje | Uso |
|----------|-----------|-----|
| Entrenamiento | 60% | Ajuste de pesos del modelo |
| Validación | 20% | Monitoreo del entrenamiento y ajuste de hiperparámetros |
| Test | 20% | Evaluación final del rendimiento real del modelo |

Esta partición 60/20/20 es importante porque garantiza que el modelo sea evaluado sobre datos que nunca vio durante el entrenamiento ni durante la validación, dando una medida honesta de su capacidad de generalización.

**Clases del dataset (11 en total):**

| # | Clase | Descripción |
|---|-------|------------|
| 1 | `healthy` | Hoja sana, sin enfermedad |
| 2 | `Early_blight` | Tizón temprano (hongo *Alternaria solani*) |
| 3 | `Late_blight` | Tizón tardío (oomiceto *Phytophthora infestans*) |
| 4 | `Leaf_Mold` | Moho de la hoja (*Passalora fulva*) |
| 5 | `Septoria_leaf_spot` | Mancha foliar por Septoria |
| 6 | `Spider_mites Two-spotted_spider_mite` | Ácaro araña de dos manchas |
| 7 | `Target_Spot` | Mancha en diana (*Corynespora cassiicola*) |
| 8 | `Tomato_Yellow_Leaf_Curl_Virus` | Virus del enrollamiento amarillo |
| 9 | `Tomato_mosaic_virus` | Virus del mosaico del tomate |
| 10 | `Bacterial_spot` | Mancha bacteriana (*Xanthomonas* spp.) |
| 11 | `powdery_mildew` | Mildiu polvoroso (oídio) |

---

## 3. Arquitecturas

Se proponen y comparan tres arquitecturas de complejidad creciente:

### 3.1 Arquitectura Base — CNN desde Cero

Una red convolucional sencilla entrenada desde cero, sin conocimiento previo. Sirve como línea base para comparar contra las arquitecturas más sofisticadas. Compuesta por bloques Conv → ReLU → MaxPool seguidos de capas densas para clasificación final en 11 clases.

### 3.2 Arquitectura Propuesta — CNN Ajustada a los Datos

Una CNN con mayor profundidad, batch normalization, dropout y data augmentation (volteos, rotaciones, ajuste de brillo) diseñada específicamente teniendo en cuenta la naturaleza de las imágenes del dataset: variabilidad de iluminación entre escenas de laboratorio y campo abierto, y diferencias sutiles entre clases como Early Blight y Late Blight.

### 3.3 Arquitectura con Transfer Learning — ResNet18 / MobileNetV2

Se utiliza una red preentrenada en ImageNet y se realiza fine-tuning reemplazando la capa final por un clasificador de 11 clases. Esta es la arquitectura con mayor rendimiento esperado dado el tamaño del dataset. MobileNetV2 se prioriza por ser un modelo liviano adecuado para despliegue en dispositivos móviles, alineándose con el objetivo del dataset original.

---

