# Desafío técnico UiFlou - Hitomi Diaconchuk

El presente repositorio contiene las resoluciones del desafío técnico para UiFlou.

A continuación se explican los detalles por cada ejercicio:

## Ejercicio 1 (Incompleto)
>Implementar uno o varios script en Python que:
>  - Procese un video MP4 para realizar:
>    ○ Pose estimation
>    ○ Detección de acciones humanas (HAR)
>  - utilizar la/las redes que considere necesarias.
>  - Las acciones a detectar tienen una de 9 segundos
>  - Implementar tracking para identificar a todas las personas presentes en el video.
>  - Calcule los ángulos del cuerpo de cada persona.
>  - Generar archivos de datos parciales cada 1 segundo, en el formato que considere óptimo.
>  - Generar un video procesado con poses, acciones y anotaciones visuales.
>  - Considerar que los datos van a serán consumidos y reproducidos con el video independientemente de en que archivo se encuentren por un proceso externo.
>  - Suba los archivos automáticamente a AWS S3

### Decisiones principales para la resolución:
- Modelo de detección de objetos: yolo11n
- Modelo de estimación de poses: Mediapipe
- Modelo de ReID: model: yolo11n-cls
- Video de prueba: https://www.youtube.com/watch?v=34rwkZQdOxI

### Puntos implementados hasta el momento:
- Tracking para identificación de personas
- Estimación de poses
- Generar video con poses y anotaciones visuales (no acciones)

Inicialmente para este ejercicio se intentó una implementación con yolo11n para la detección de objetos y Mediapipe para la estimación de poses:
decisión se basó en el hecho de que ambos son dos de los modelos mas precisos en su categoría y es común verlos implementados de esta forma en la industria.

Hasta el momento el mayor desafío fue configurar el modelo para mantener una id consistente sobre cada sujeto identificado. Para esto tengo la carpeta "tracker_configurations"
en la que guardo el archivo de configuración del tracker con los parámetros que mejores resultados me permitieron obtener en cada video de prueba. Para el video de prueba listado
anteriormente que es particularmente caótico se resaltan las siguientes configuraciones:

- Selección de BoT-SORT como tracker para poder activar ReID.
- Un new_track_thresh relativamente alto para desalentar nuevos ID.
- Un track_buffer alto para mantener las ID's perdidas durante mas cuadros.
- Un IoU mínimo bajo para compensar la poca superpocisión del área de identificación debido al movimiento de la cámara.
- Un threshold de apariencia relativamente bajo para una re-identificación mas laxa.
- Selección de yolo11n-cls como modelo de ReID (demostró una capacidad de identificación superior a la opción por defecto el cual es el mismo modelo de detección).

Para el resto de los puntos del enunciado se propone lo siguiente:

- Generar datos parciales en formato JSONL para facilitar su consumo por procesos externos.
- Procesar los 9 segundos de acción en N cuadros con N = input_fps * 9. Si se utiliza el salto de cuadros en la etapa de inferencia anterior no hace falta procesar el 100% de N.
- Adoptar una estructura de pipeline con subprocesos (similar a la implementada en el ejercicio 2) para obtener un mejor rendimiento.


## Ejercicio 2 (Completo)
>Se requiere aplicar pose estimación y object detection a una cámara RTSP y obtener video procesado como resultado
>que debe poder ser consumido durante la captura o finalizada la misma.
>  - Definir estructura del sistema
>  - Definir el formato de video
>  - Definir formato de datos
>  - Implementar uno o varios scripts python para resolver la problemática.
>  - Almacenar los datos de salida donde lo considere pertinente.

### Decisiones principales para la resolución:
- Modelo de estimación de poses y detección de objetos: yolo11n-pose.
- Formato de video: fragmentos secuenciales de 150 frames en MP4 con framerate y resolución original.
- Formato de datos: JSONL.
- Almacenamiento de datos de salida: local.
- Patrones de diseño: Productor-Consumidor.
- Video de prueba: https://www.youtube.com/watch?v=34rwkZQdOxI (igual que en el anterior).

Con lo aprendido en el ejercicio anterior, esta vez opté por utilizar yolo11n-pose para poder realizar tanto la detección de objetos como la estimación de poses.
A diferencia de otros modelos, yolo-pose permite realizar ambas inferencias en un único llamado y cuenta con una estimación de poses particularmente rápida pero que demostró ser lo suficientemente
confiable en el entorno caótico del video de prueba.

El desafío de este ejercicio estuvo en mantener un rendimiento lo suficientemente rápido como para no atrasarse significativamente con respecto al video que se estaría transmitiendo en tiempo real.
La transmisión de prueba fue de 25 FPS y la capacidad inicial para procesar fue de 7~8 FPS.
Para intentar mejorar el rendimiento se implementó una estrategia de salto de cuadros aplicando una inferencia cada una determinada cantidad de cuadros capturados. Con un salto de 1 cuadro (que se traduce al 50% de los cuadros capturados)
se logró llegar a un rendimiento de alrededor de 12 FPS.
Finalmente, se implementó un patrón Productor-Consumidor en el cual se define un productor en un subproceso encargado de obtener cada cuadro de la transmisión en vivo y almacenarlos en una cola a
disposición del bucle principal de procesamiento (consumidor) que realiza las inferencias en paralelo utilizando CUDA.

De esta forma, el proceso secuencial que inicialmente consistía en:
> Captar cuadro -> Inferencia -> Anotación de cuadro -> Escritura de cuadro

Se transforma en la siguiente pipeline:
> Productor(Captar fotogramas) -> Cola de fotogramas -> Consumidor(procesamiento) -> Almacenamiento

Así, la ejecución del procesamiento de un cuadro o de la captura de un cuadro ya no se limitan entre si por la secuencialidad de estar en un mismo proceso. Esto permitió elevar el rendimiento
a un rango de 15 a 20 FPS dependiendo de la complejidad de cada inferencia: mas que suficiente para lograr procesar los 25 FPS (de los cuales se procesan el 50%, por lo tanto representan una carga real de 12.5 FPS) de la entrada en
tiempo real.

Para mejorar aún más el rendimiento se evalúa la posibilidad de identificar otros pasos que ralenticen la ejecución secuencial del bucle (muy probablemente el dibujado sobre cuadros y la escritura en disco) y ejecutarlos en subprocesos.

Finalmente, para permitir el consumo de la salida por procesos externos y según se esté generando, se optó por generar el video en segmentos de N cuadros (variable configurable actualmente establecida en 150) y por almacenar los datos en un
archivo de tipo JSONL con la siguiente estructura:
'''
frame_data = {
            "frame": frame_number,
            "objects": [],
            "keypoints": []
        }
'''

El archivo JSONL es un estandar universal compatible con todo tipo de aplicaciones y la segmentación del video permite que se vayan ejecutando a medida que se van generando casi en tiempo real.
Para una reproducción de video en tiempo real se evalúa implementar la transmisión cuadro a cuadro en el bucle de procesamiento.
