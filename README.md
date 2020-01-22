# Face_Recognizer

## Real time recognizer using Insightface.

### Introducción
###### Con el estallido del Machine Learning y el análisis de datos ha cobrado mucha fuerza el reconocimiento facial por camara de video en tiempo real en aplicaciones de monitoreo, vigilancia y hasta en accesibilidad y seguridad.
###### Este software es capaz de realizar la detección de todos los rostros en una imagen o video, y, extrayendo ciertas características biométricas de estos los compara con los datos biométricos que se tienen una base de datos de personas registradas para realizar la identificación de las personas detectadas.

#### Requisitos e Instalación
Python >=3.5
Numpy
OpenCV >=4

###### Apache MXNet
https://mxnet.apache.org/get_started/?version=v1.5.1&platform=linux&language=python&environ=pip&processor=gpu

>>pip3 install mxnet-cuXXX

Nota: el numero al final es la version de cuda que se va a instalar. si es cuda 10.1 se instala con pip3 install mxnet-cu101

###### Insightface
https://github.com/deepinsight/insightface

>>pip3 install insightface
###### En caso de desear desde el código fuente para entrenar un modelo propio seguir las instrucciones en la página.
###### También en caso de preferir emplear otra librería de deep learning como tensorflow, pytorch, caffe, etc seguir los links al final.

### Marco teórico
###### EXP. Se tienen 2 problemas a resolver: Detección de rostros en una imagen y Reconocimiento de las personas detectadas. En este caso se hará uso del detector y reconocedor de Insightface.
### Detección
###### El algoritmo de detección de rostros que se emplea es llamada RetinaFace. en este algoritmo se realiza un análisis por pixeles de la imagen para detectar los rostros a diferentes escalas y se emplea una arquitectura (CNN) ResNet50 [2].
### Reconocimiento
###### Para la detección se obtiene características de los rostros y se entrena el modelo (CNN) con una gran cantidad de datos y se utiliza una función de costo (Loss) para determinar el error a reducir en el entrenamiento. Insightface utiliza una función de costo llamada ArcFace (Additive Angular Margin Loss) [3].
##### Las detecciones generan un código (Embedding) los cuales deben ser lo más similar (Similaridad Vectorial) al código de un rostro de la misma persona en tiempo y condiciones distintas, y a su vez debe ser lo más diferente posible del código generado para otras personas.

### Implementación

#### Registro
###### La detección de rostros es un algoritmo que puede funcionar de forma inmediata (Después de entrenar el modelo claro está) pues al ejecutarlo este detectará los rostros que hay en una imagen y ya está, sin embargo para realizar el reconocimiento de personas es necesario comparar los sujetos que deseamos identificar con personas de las que tenemos información y etiquetarlos, y en caso de que no coincida con ninguna la etiquete como desconocida y si es necesario se guarda su información para un futuro. El algoritmo de registro simplemente se toma una cantidad de fotos de un usuario para agregar a la base de datos. Mediante una reproducción en vivo del sujeto se toman una cantidad de fotos, en estas fotos se realiza detección de rostros, luego se obtienen los encodings de estas fotos y se guardan en la base de datos perteneciendo al usuario registrado.

#### Reconocimiento
###### Tomando imagen en vivo de video en cada frame se realiza detección de rostros, luego, de cada uno de los rostros detectados se segmentan, se re escalan a un tamaño fijo, y se obtienen los encodings para luego ser comparados con los encodings registrados en la base de datos. El cálculo de la similitud se puede realizar de varias maneras, pero en este algoritmo se emplea la similitud vectorial, que es la distancia entre 2 vectores normalizados.

###### Calculada la similitud se puede etiquetar con quien haya obtenido la mayor, siempre y cuando se haya superado el umbral, en caso contrario se toma como desconocido. Un método más confiable (y el que se emplea aquí) es tomar el nombre del que más aparece con la mayor similitud, haciendo uso de múltiples códigos por sujeto y aumentando la fiabilidad.


### Ejecución
###### Nota: Esta implementación emplea GPU tanto para la detección como para el reconocimiento. es posible que la GPU no sea capaz de cargar ambos modelos, o que no se disponga de GPU en absoluto, para esto se debe modificar en el archivo processing_dataset.py en las líneas 111 (detección) y 115 (reconocimiento) ctx_id de 0 a -1 para que funcione con CPU solamente, y en el archivo face_recognition_v2.py en las líneas 181 y 185. 

#### Registro
###### Se tienen 2 códigos con los que se realiza el reconocimiento, uno de registro de personas register.py, donde se pasa como parámetro el nombre del usuario a registrar y se inicializa una secuencia de video por cámara.

>>python3 register.py -n [Nombre]

###### En esta secuencia de video se toma una fotografía del usuario con la tecla “S”. Se deben tomar (Faltan más pruebas para un número óptimo) unas 6 fotos. Para finalizar se presiona la tecla “Q” con lo que se cierra el video y se procesarán las fotos registradas, se obtienen los códigos y se agregan a la base de datos (Por ahora es solo un archivo. en un futuro se puede usar de una base de datos como MySQL o PostgreSQL).

#### Reconocedor
###### El archivo face_recognition.py analiza en video real las personajes que aparecen en cámara y las etiqueta según la información que se tenga guardada en la base de datos.
###### El archivo face_recognition_v2.py funciona similar pero a la hora de obtener la similitud toma en cuenta la etiqueta que más aparece entre las múltiples imágenes guardadas. Probablemente más confiable.
###### Se ejecuta con el comando:
>>python3 face_recognition_v2.py

### Líneas a Futuro
###### Implementación del algoritmo en una aplicación específica. un caso podría ser en monitoreo de personas en un lugar, donde se pueden registrar automáticamente sujetos desconocidos para examinar su comportamiento e identificarlos, ya sea en un almacén, o en sistemas de cámaras de seguridad.
###### Aplicar el sistema de forma que no requiera registro manual, si no que en su lugar tome los sujetos nuevos y los guarde para identificarlos a futuro en caso de ser recurrentes en un lugar.
###### Implementación de Anti-spoofing. Actualmente este algoritmo no es capaz de diferenciar si se trata de una fotografía, un video pregrabado o un video en vivo. implementando un sistema de anti-spoofing se podría implementar en sistemas de seguridad de acceso vía registro de rostro.

