# T2S GPT

Instancia independiente del proceso de TIPS GPT para Target2-Securities.

El producto no usa `trilemma-simulator` ni comparte datos con `tips/`. Su corpus vive en `t2sgpt/data/`, su salida en `t2sgpt/output/` y sus secretos en `t2sgpt/secrets/`.

## Flujo rapido

```powershell
cd C:\Users\Jose-Firebat\proyectos\trilemma\t2sgpt
python -m pip install -r requirements.txt
python t2s_ingest.py
python t2s_ask.py "Explica el ciclo de liquidacion DvP en T2S" --context
python t2s_web.py
```

La web queda en `http://127.0.0.1:8790/T2S/` salvo que cambies `T2S_WEB_PORT`.

## Ingesta

`t2s_ingest.py` rastrea la pagina ECB Professional Use de T2S:

```text
https://www.ecb.europa.eu/paym/target/target-professional-use-documents-links/t2s/html/index.en.html
```

Tambien sigue, de forma acotada, subpaginas profesionales enlazadas de T2S, documentacion compartida `coco` y la pagina ECB de change requests T2S. Descarga PDF/HTML/XLSX/ZIP, extrae texto y construye un indice local hibrido TF-IDF + char TF-IDF + BM25.

Para una prueba corta:

```powershell
python t2s_ingest.py --limit 12 --max-pages 4
```

Para reconstruir:

```powershell
python t2s_ingest.py --force
```

Por defecto se recorren hasta 128 paginas profesionales enlazadas. Puedes subirlo con `--max-pages` o `T2S_MAX_CRAWL_PAGES` si el ECB amplia la estructura.

## Web y acceso

Variables principales:

```powershell
$env:T2S_WEB_HOST="0.0.0.0"
$env:T2S_WEB_PORT="8790"
$env:T2S_AUTH_DISABLED="true"
$env:T2S_PUBLIC_BASE_URL="https://TU-DOMINIO"
$env:T2S_SESSION_SECRET="pon-un-secreto-largo-aleatorio"
```

Gestion de usuarios:

```powershell
python access_admin.py approve usuario@example.com --name "Usuario"
python access_admin.py list
```

## Watcher

```powershell
python t2s_doc_watcher.py --check-only --verbose
```

El watcher compara el Professional Use actual con el corpus procesado y, si se ejecuta sin `--check-only`, reconstruye con `t2s_ingest.py --refresh-index`.

## Publicacion

El repositorio debe contener solo codigo y configuracion. El corpus (`data/`), salidas (`output/`) y secretos (`secrets/`) se generan en cada entorno y estan ignorados por Git.

Cuando exista `trilemmaconsulting/t2sgpt` y la cuenta tenga permiso de escritura:

```powershell
git remote add origin https://github.com/trilemmaconsulting/t2sgpt.git
git push -u origin main
```

En produccion, tras clonar:

```powershell
python -m pip install -r requirements.txt
python t2s_ingest.py --force
python qa_t2s.py
python t2s_web.py
```
