# Clasificación de Enfermedades en Hojas de Tomate mediante Redes Neuronales Convolucionales
 
---
 
## 1. Problema a Resolver
 
El tomate (*Solanum lycopersicum*) es uno de los cultivos hortícolas más importantes de Colombia. Según datos de Bayer Agro Colombia, el país siembra aproximadamente **9.000 hectáreas** de tomate y produce alrededor de **512.000 toneladas al año**, con un rendimiento promedio de 62,3 toneladas por hectárea. El cultivo está presente en al menos 21 departamentos, siendo Boyacá, Caldas, Risaralda y Cundinamarca los principales productores. Las variedades más cultivadas son el Milano y el Chonto, ampliamente consumidas en la cocina colombiana.
 
A pesar de su relevancia económica, el cultivo de tomate es altamente vulnerable a enfermedades foliares. Cuando una hoja de tomate se ve afectada por patógenos como hongos, bacterias o virus, las consecuencias no se limitan a la hoja en sí: la planta pierde capacidad fotosintética, lo que reduce directamente el tamaño, la calidad y el número de frutos producidos. Enfermedades como el tizón tardío (*Late Blight*) —la misma que devastó los cultivos de papa en Irlanda en el siglo XIX— pueden arrasar un cultivo completo en cuestión de días si no se detectan a tiempo. Otras como el virus del enrollamiento amarillo (*Tomato Yellow Leaf Curl Virus*) no tienen cura una vez establecidas, por lo que la detección temprana es la única estrategia efectiva.
 
El problema central es que la mayoría de los agricultores colombianos, especialmente los pequeños productores, no tienen acceso inmediato a un agrónomo experto. El diagnóstico visual de enfermedades requiere formación especializada y tiempo, y cuando finalmente se identifica el problema, la enfermedad ya puede haber avanzado significativamente. Esto se traduce en pérdidas económicas directas para el agricultor y en desperdicio de alimentos que afectan el abastecimiento regional.
 
**¿Qué proponemos hacer?** Desarrollar y evaluar un modelo de clasificación de enfermedades en hojas de tomate basado en redes neuronales convolucionales. El modelo recibe como entrada una imagen de una hoja de tomate y predice a cuál de las 11 clases pertenece (10 enfermedades o hoja sana). El alcance de este proyecto es el entrenamiento, validación y evaluación del modelo; el resultado es un modelo entrenado y listo para ser usado en trabajos futuros, ya sea integrándolo en una aplicación o utilizándolo para clasificar nuevas imágenes directamente.
 
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
 
Se proponen y comparan tres arquitecturas de complejidad creciente. Los diagramas de cada una están disponibles en los archivos `.drawio` adjuntos, listos para abrir en [draw.io](https://app.diagrams.net).
 
### 3.1 Arquitectura Base — CNN desde Cero
 
Red convolucional sencilla entrenada desde cero, sin conocimiento previo. Funciona como **línea base** para medir cuánto mejoran las arquitecturas más avanzadas.
 
Flujo: `Input (3×224×224)` → 3 bloques de `Conv2d → ReLU → MaxPool2d` con 32, 64 y 128 filtros respectivamente → `Flatten` → `Linear 512 + ReLU + Dropout(0.5)` → `Salida 11 clases`
 
### 3.2 Arquitectura Propuesta — CNN Mejorada
 
CNN más profunda diseñada específicamente para este dataset, que combina imágenes de laboratorio y campo abierto. Se añaden BatchNorm para estabilizar el entrenamiento, Dropout para reducir overfitting y Data Augmentation (volteos, rotaciones, ajuste de brillo y contraste) para mejorar la generalización.
 
Flujo: `Input` → `Data Augmentation` → 4 bloques de `Conv2d → BatchNorm2d → ReLU → MaxPool2d` con 32, 64, 128 y 256 filtros → `Flatten` → `Linear 512 + BN + ReLU + Dropout(0.4)` → `Linear 256 + ReLU + Dropout(0.3)` → `Salida 11 clases`
 
### 3.3 Arquitectura con Transfer Learning — MobileNetV2
 
Red preentrenada en ImageNet con 1.2 millones de imágenes. El backbone se congela (sus pesos no se modifican) y se reemplaza únicamente la capa de clasificación final por una nueva cabeza de 11 clases. MobileNetV2 se elige por ser un modelo liviano, ideal para despliegue en dispositivos móviles sin conexión a internet.
 
Flujo: `Input` → `MobileNetV2 Feature Extractor (congelado)` → `AdaptiveAvgPool2d (1280 features)` → `Linear 512 + BN + ReLU + Dropout(0.3)` → `Salida 11 clases`
 
---
