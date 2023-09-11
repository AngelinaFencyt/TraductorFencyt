## SIGNA : TRADUCTOR DE LENGUAJE DE SEÑAS PERUANO

Nuestra solución se basa en una arquitectura única de codificador-decodificador. El codificador es una versión significativamente mejorada de Squeezeformer, en la que la extracción de características se ha adaptado para manejar puntos de referencia de la tubería media en lugar de señales del habla. El descodificador es un simple transformador de dos capas. Además, predijimos una puntuación de confianza para identificar ejemplos corruptos que pueden ser útiles para el postprocesamiento. También introdujimos aumentos eficientes y creativos para regularizar el modelo, siendo los más importantes CutMix, FingerDropout y TimeStretch, DecoderInput Masking. Utilizamos pytorch para desarrollar y entrenar nuestros modelos y luego tradujimos manualmente la arquitectura del modelo y portamos los pesos a tensorflow, desde donde exportamos a tf-lite.

![](architecture_overview.png)







## PREPARACIÓN

Utilizamos el  `nvcr.io/nvidia/pytorch:23.07-py3` contendedor del [ngc catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) para tener un entorno coherente entre los miembros del equipo. Puede ejecutarlo a través de:
`docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.07-py3`

Dentro del contenedor clone este repositorio e instale los paquetes necesarios con ```
git clone https://github.com/AngelinaFencyt/TraductorFencyt.git
cd kaggle-asl-fingerspelling-1st-place-solution
pip install -r requirements.txt
```

```


Por defecto el entrenamiento se registra a través de neptune.ai en un proyecto quickstart. Si quieres usar tu propio proyecto neptune establece `cfg.neptune_project` en `configs/cfg_1.py` y `configs/cfg_2.py`. 

      
### 1. Train round 1

Train 4 folds of cfg_1:

```
python train.py -C cfg_1
python train.py -C cfg_1 --fold 1
python train.py -C cfg_1 --fold 2
python train.py -C cfg_1 --fold 3
```

Add oof predictions from step 1. to train_folded.csv and concatenate with supplemental metadata:

```
python scripts/get_train_folded_oof_supp.py 
```

### 2. Train round 2

Train 2x fullfit seeds of cfg_2:

```
python train.py -C cfg_2 --fold -1
python train.py -C cfg_2 --fold -1
```

### 3. TF-Lite conversion

Transfer the resulting weights to a tensorflow equivalent ensemble model and export to tf-lite:

```
python scripts/convert_cfg_2_to_tf_lite.py  
```


The final model is saved under

```
datamount/weights/cfg_2/fold-1/model.tflite 
datamount/weights/cfg_2/fold-1/inference_args.json
```
and can be added to a kaggle kernel and submitted.


## References

We adapted squeezeformer components from these two great repositories: 

- SqueezeFormer (tensorflow) https://github.com/kssteven418/Squeezeformer
- SqueezeFormer (pytorch) https://github.com/upskyy/Squeezeformer/

Check out the SqueezeFormer [paper](https://arxiv.org/pdf/2206.00888.pdf) for more details on the architecture.

We copied and adapted the TFSpeech2TextDecoder from https://github.com/huggingface/transformers/ to support caching and used components related to LLama Attention.

## Paper 

TBD
      
      
      
# SegundoFencyt
# TraductorFencyt
