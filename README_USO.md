# Sistema simplificado MSA

Este paquete contiene solo los scripts necesarios para usar el sistema simplificado reportado. No incluye audios, features, metricas, figuras ni archivos del informe.

## Entorno

```bash
conda env create -f environment.yml
conda activate codex_msa_features
```

## Datos esperados

El sistema asume esta estructura relativa a la carpeta donde se ejecuta:

```text
feature_outputs/rwc_p_100_orthogonal_v1/*_features.npz
data/rwc-annotations-archive/
```

Si se parte desde audio crudo, primero hay que generar las features con `batch_extract_msa_features.py` / `extract_msa_features.py`.

## Orden de uso

Para correr exactamente la rama simplificada desde cero, primero deben existir las predicciones fuente y la tabla de candidatos. El orden práctico es:

```bash
python run_disruptive_ideas_experiments.py --limit 100 --folds 5
python run_ad01_candidate_rescue_iteration.py --limit 100 --folds 5
python run_adres_simplification_iteration.py --limit 100 --folds 5
python run_adres_minimal_model_iteration.py --limit 100 --folds 5
```

La salida final del sistema mínimo queda en:

```text
structure_outputs/rwc_p_100_adres_minimal_models_v1/
```

El archivo principal de resultados es:

```text
structure_outputs/rwc_p_100_adres_minimal_models_v1/aggregate_metrics.csv
```

## Script principal

El script que implementa el sistema mínimo reportado es:

```text
run_adres_minimal_model_iteration.py
```

Los demas `run_*.py` incluidos son dependencias locales necesarias para construir las predicciones fuente, la fusion AD01/ADRES y las utilidades de evaluacion.
