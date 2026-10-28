#!/usr/bin/env python3
"""Casa as relacoes de sample do WhoSampled com o dump local do MusicBrainz.

Para cada relacao do dataset do WhoSampled, busca os dois lados (a musica e a
musica sampleada) no Postgres com busca fuzzy por trigram, filtra os candidatos
com difflib e grava o resultado reduzido em lotes .jsonl.gz. Cada lado é buscado
sob TODOS os artistas creditados na faixa, e os resultados dos varios artistas
entram misturados na mesma lista.

Modos:
  (padrao)  minera o dataset inteiro, em lotes e retomavel
  --probe   consulta uma unica faixa e imprime o JSON (debug)
  --stats   resume o que ja foi gravado no diretorio de saida

Exemplos:
  # roda tudo com os defaults
  scripts/mine_musicbrainz.py

  # teste rapido: 500 relacoes num diretorio separado
  scripts/mine_musicbrainz.py --max-items 500 --output-dir /tmp/teste

  # afrouxa o corte fuzzy e guarda mais releases por match
  scripts/mine_musicbrainz.py --similarity-threshold 0.45 --max-releases 50

  # debug de uma faixa so (o ULTIMO argumento é a musica)
  scripts/mine_musicbrainz.py --probe "The Winstons" "Amen, Brother"

  # ... com varios artistas creditados
  scripts/mine_musicbrainz.py --probe "Jarren Benton" "R.A. The Rugged Man" "Smells Like"

  # progresso do run atual
  scripts/mine_musicbrainz.py --stats
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import psycopg2
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
DEFAULT_INPUT = REPO_ROOT / "data" / "whosampled-data-samples.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "musicbrainz_whosampled_final"


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
# A query original partia de `recording.name %% <track>`, o que fazia o planner
# varrer o indice trigram de recordings inteiro (~583k linhas de indice +
# recheck de ~6M linhas no heap) antes de qualquer filtro de artista.
#
# Aqui a ordem é invertida: primeiro os artistas parecidos (indice trigram
# pequeno e seletivo), depois os artist_credits deles, e só entao as recordings
# desses creditos -- o filtro de nome roda sobre alguns milhares de linhas em vez
# de 30M. O MATERIALIZED é essencial: sem ele o planner volta a inlinear as CTEs
# e escolhe o plano lento.
#
# O payload JSON (tags, data de lancamento, releases) é montado apenas para as
# --max-results melhores linhas, e nao para todos os matches intermediarios.
#
# Os campos do artista (type, gender, area, begin_area, end_area) NAO precisam de
# subquery: `row_to_json(a)` ja traz as colunas da propria linha de artist. Os ids
# viram nome no Python, pelo AreaResolver.
#
# ATENCAO sobre datas: as views derivadas `recording_first_release_date` e
# `release_first_release_date` existem no dump mas estao VAZIAS (0 linhas), entao
# qualquer data tirada dali sai sempre NULL. As datas abaixo vem das tabelas
# base: `release_country` + `release_unknown_country` (a data de um release é a
# menor entre as duas) e `release_group_meta.first_release_date_*`.
#
# Mapa das colunas do resultado -- `extract_musicbrainz_metadata` le as linhas
# por posicao, entao inserir coluna no meio quebra o downstream:
#
#   0 artist_data     3 tag_recording_data  6 tag_release_data        9 release_group_data
#   1 credit_data     4 tag_artist_data     7 tag_release_group_data 10 match_score
#   2 recording_data  5 release_date        8 release_data

QUERY_COMPLEX = """
WITH artist_match AS MATERIALIZED (
    SELECT a.id AS artist_id,
           similarity(a.name, %(artist)s) AS artist_sim
    FROM musicbrainz.artist a
    WHERE a.name %% %(artist)s
),
credit_match AS MATERIALIZED (
    SELECT acn.artist_credit AS credit_id,
           am.artist_id,
           am.artist_sim
    FROM artist_match am
    JOIN musicbrainz.artist_credit_name acn ON acn.artist = am.artist_id
),
recording_candidate AS MATERIALIZED (
    SELECT r.id AS recording_id, r.name, r.artist_credit AS credit_id
    FROM musicbrainz.recording r
    WHERE r.artist_credit = ANY (ARRAY(SELECT DISTINCT credit_id FROM credit_match))
),
matched AS (
    SELECT DISTINCT ON (cm.artist_id, rc.recording_id)
           cm.artist_id,
           rc.credit_id,
           rc.recording_id,
           cm.artist_sim + similarity(rc.name, %(track)s) AS match_score
    FROM recording_candidate rc
    JOIN credit_match cm ON cm.credit_id = rc.credit_id
    WHERE rc.name %% %(track)s
    ORDER BY cm.artist_id, rc.recording_id,
             cm.artist_sim + similarity(rc.name, %(track)s) DESC
),
top_matches AS (
    SELECT * FROM matched ORDER BY match_score DESC LIMIT %(max_results)s
)
SELECT
    row_to_json(a)  AS artist_data,
    row_to_json(ac) AS credit_data,
    row_to_json(r)  AS recording_data,

    (SELECT json_agg(row_to_json(tr))
     FROM musicbrainz.recording_tag rt
     JOIN musicbrainz.tag tr ON tr.id = rt.tag
     WHERE rt.recording = r.id
    ) AS tag_recording_data,

    (SELECT json_agg(row_to_json(ta))
     FROM musicbrainz.artist_tag att
     JOIN musicbrainz.tag ta ON ta.id = att.tag
     WHERE att.artist = a.id
    ) AS tag_artist_data,

    -- Data de lancamento mais antiga entre TODOS os releases em que a recording
    -- aparece. Antes isto lia `recording_first_release_date`, que esta vazia no
    -- dump -- o campo saia sempre NULL. O formato de saida (lista com um objeto
    -- {year, month, day}) é o mesmo de antes.
    (SELECT json_agg(row_to_json(x))
     FROM (
        SELECT d.date_year AS year, d.date_month AS month, d.date_day AS day
        FROM (SELECT DISTINCT md.release AS release_id
              FROM musicbrainz.track tk
              JOIN musicbrainz.medium md ON md.id = tk.medium
              WHERE tk.recording = r.id) rr
        CROSS JOIN LATERAL (
            SELECT rc.date_year, rc.date_month, rc.date_day
            FROM musicbrainz.release_country rc WHERE rc.release = rr.release_id
            UNION ALL
            SELECT ruc.date_year, ruc.date_month, ruc.date_day
            FROM musicbrainz.release_unknown_country ruc WHERE ruc.release = rr.release_id
        ) d
        WHERE d.date_year IS NOT NULL
        ORDER BY d.date_year, d.date_month NULLS LAST, d.date_day NULLS LAST
        LIMIT 1
     ) x
    ) AS release_date,

    -- Tags dos releases. As tags de recording/artist sao esparsas no
    -- MusicBrainz; o sinal de genero mora um nivel acima, no release e no
    -- release_group. Caminho: recording <- track -> medium -> release.
    -- O DISTINCT no release é necessario: sem ele uma faixa que aparece em
    -- dois mediums do mesmo release conta a tag em dobro.
    (SELECT json_agg(row_to_json(x))
     FROM (
        SELECT t.name, SUM(rt2.count) AS count
        FROM (SELECT DISTINCT md.release AS release_id
              FROM musicbrainz.track tk
              JOIN musicbrainz.medium md ON md.id = tk.medium
              WHERE tk.recording = r.id) rr
        JOIN musicbrainz.release_tag rt2 ON rt2.release = rr.release_id
        JOIN musicbrainz.tag t           ON t.id = rt2.tag
        GROUP BY t.name
        ORDER BY SUM(rt2.count) DESC
     ) x
    ) AS tag_release_data,

    -- Idem para os release_groups. Aqui o DISTINCT importa ainda mais: dezenas
    -- de releases (reedicoes, edicoes regionais) apontam para o mesmo grupo, e
    -- sem ele a contagem da tag sairia inflada -- medido em 'When a Man Loves a
    -- Woman' (496 releases / 404 grupos): soul 233 com DISTINCT, 330 sem.
    (SELECT json_agg(row_to_json(x))
     FROM (
        SELECT t.name, SUM(rgt.count) AS count
        FROM (SELECT DISTINCT rl.release_group AS rg_id
              FROM musicbrainz.track tk
              JOIN musicbrainz.medium md  ON md.id = tk.medium
              JOIN musicbrainz.release rl ON rl.id = md.release
              WHERE tk.recording = r.id) gg
        JOIN musicbrainz.release_group_tag rgt ON rgt.release_group = gg.rg_id
        JOIN musicbrainz.tag t                 ON t.id = rgt.tag
        GROUP BY t.name
        ORDER BY SUM(rgt.count) DESC
     ) x
    ) AS release_group_tag_data,

    -- Identidade dos releases, mais antigos primeiro e limitada a
    -- --max-releases. Serve so para voltar ao MusicBrainz depois -- as tags
    -- acima ja cobrem todos os releases, inclusive os cortados aqui.
    (SELECT json_agg(row_to_json(x))
     FROM (
        SELECT DISTINCT rl.gid, rl.name, ry.year
        FROM musicbrainz.track tk
        JOIN musicbrainz.medium md  ON md.id = tk.medium
        JOIN musicbrainz.release rl ON rl.id = md.release
        LEFT JOIN LATERAL (
            SELECT MIN(dd.date_year) AS year
            FROM (
                SELECT rc.date_year FROM musicbrainz.release_country rc
                WHERE rc.release = rl.id
                UNION ALL
                SELECT ruc.date_year FROM musicbrainz.release_unknown_country ruc
                WHERE ruc.release = rl.id
            ) dd
        ) ry ON TRUE
        WHERE tk.recording = r.id
        ORDER BY ry.year NULLS LAST
        LIMIT %(max_releases)s
     ) x
    ) AS release_data,

    -- Release groups + tipo primario (Album / Single / EP / Compilation /
    -- Soundtrack) + ano da primeira edicao. Sem limite: sao poucos por
    -- recording. Os LEFT JOIN sao porque release_group.type é nullable e nem
    -- todo grupo tem meta com data.
    (SELECT json_agg(row_to_json(x))
     FROM (
        SELECT DISTINCT rg.gid, rg.name, rgpt.name AS primary_type,
               rgm.first_release_date_year AS year
        FROM musicbrainz.track tk
        JOIN musicbrainz.medium md        ON md.id = tk.medium
        JOIN musicbrainz.release rl       ON rl.id = md.release
        JOIN musicbrainz.release_group rg ON rg.id = rl.release_group
        LEFT JOIN musicbrainz.release_group_primary_type rgpt ON rgpt.id = rg.type
        LEFT JOIN musicbrainz.release_group_meta rgm          ON rgm.id = rg.id
        WHERE tk.recording = r.id
     ) x
    ) AS release_group_data,

    tm.match_score

FROM top_matches tm
JOIN musicbrainz.artist a         ON a.id  = tm.artist_id
JOIN musicbrainz.artist_credit ac ON ac.id = tm.credit_id
JOIN musicbrainz.recording r      ON r.id  = tm.recording_id
ORDER BY tm.match_score DESC;
"""


# ---------------------------------------------------------------------------
# Hierarquia de areas
# ---------------------------------------------------------------------------
# `artist.area` (e `begin_area` / `end_area`) aponta pra qualquer nivel da
# hierarquia geografica: das ~1,47M areas de artista, so 76% sao pais -- o resto é
# cidade (269k), subdivisao (71k), municipio, distrito... Em `begin_area` é pior
# ainda: de 490k preenchidos, so 60k sao pais. Gravar esse nome cru como "country"
# fazia a analise descartar o registro (o filtro por lista de paises nao casa com
# "Queensbridge").
#
# A hierarquia mora em `l_area_area`, onde **entity1 é parte de entity0** -- ou
# seja, pra subir de nivel vai-se de entity1 (a parte) pra entity0 (o todo).
# Medido neste dump: as 120.011 linhas da tabela sao TODAS do link type 'part of'
# (o filtro por link_type abaixo é rede de seguranca, nao necessidade).
#
# Regra: se a area ja é `type = 1` (Country), é ela mesma; senao sobe ate achar
# uma. O `WHERE up.type IS DISTINCT FROM 1` no ramo recursivo é o que implementa a
# parada -- e importa: 30 paises aparecem como entity1 (territorios dependentes),
# entao sem ela Puerto Rico viraria United States.
#
# Numeros do dump: 120.025 das 120.050 areas chegam a um pais; profundidade maxima
# 7 (Queensbridge -> United States sao 4 saltos); 4 areas alcancam DOIS paises
# (territorios disputados: Golan Heights, Ramot, Beitar Illit, Bik'at HaYarden) --
# o DISTINCT ON resolve pela menor profundidade e depois pelo menor id, pra saida
# deterministica; 19 areas nunca chegam a pais (a ilha "Ireland", Jerusalem, norte
# de Chipre), e nesses casos o campo de pais sai None -- o nome cru continua no
# campo de area, entao nada se perde.

AREA_MAX_DEPTH = 12  # profundidade real é 7; a folga tambem trava ciclo no dump

AREA_COUNTRY_QUERY = """
WITH RECURSIVE up AS (
    SELECT a.id AS area_id, a.id AS cur_id, a.type, 0 AS depth
    FROM musicbrainz.area a
  UNION ALL
    SELECT up.area_id, parent.id, parent.type, up.depth + 1
    FROM up
    JOIN musicbrainz.l_area_area laa ON laa.entity1 = up.cur_id   -- entity1 é parte de
    JOIN musicbrainz.link lk         ON lk.id = laa.link
    JOIN musicbrainz.link_type lt    ON lt.id = lk.link_type AND lt.name = 'part of'
    JOIN musicbrainz.area parent     ON parent.id = laa.entity0   -- ... entity0
    WHERE up.type IS DISTINCT FROM 1 AND up.depth < %(max_depth)s
)
SELECT DISTINCT ON (up.area_id) up.area_id, c.name
FROM up
JOIN musicbrainz.area c ON c.id = up.cur_id
WHERE c.type = 1
ORDER BY up.area_id, up.depth, c.id;
"""


class AreaResolver:
    """Traduz os ids da linha de artist em nomes, sem tocar no banco durante o loop.

    Tudo o que é preciso é pequeno (120k areas, 9 tipos de area, 6 de artista, 5 de
    genero), entao carrega-se o mapa inteiro na largada -- ~1s -- e o cache ja nasce
    100% quente. A alternativa (uma query recursiva por area nova, memoizada) faria
    milhares de idas ao banco durante a mineracao pra economizar algumas dezenas de
    MB de RAM.
    """

    def __init__(self, cur, max_depth: int = AREA_MAX_DEPTH, quiet: bool = False):
        t0 = time.perf_counter()

        cur.execute("SELECT id, name, type FROM musicbrainz.area;")
        rows = cur.fetchall()
        self.area_name = {r[0]: r[1] for r in rows}
        self.area_type_id = {r[0]: r[2] for r in rows}

        cur.execute("SELECT id, name FROM musicbrainz.area_type;")
        self.area_type_name = dict(cur.fetchall())

        cur.execute("SELECT id, name FROM musicbrainz.artist_type;")
        self.artist_type = dict(cur.fetchall())

        cur.execute("SELECT id, name FROM musicbrainz.gender;")
        self.gender = dict(cur.fetchall())

        cur.execute(AREA_COUNTRY_QUERY, {"max_depth": max_depth})
        self.country = dict(cur.fetchall())

        dt = time.perf_counter() - t0
        if not quiet:
            print(f"hierarquia de areas: {len(self.area_name):,} areas | "
                  f"com pais: {len(self.country):,} | {dt:.1f}s", file=sys.stderr)

    def area(self, area_id):
        """(nome cru, nome do tipo, pais) da area -- (None, None, None) se nao houver."""
        if area_id is None:
            return None, None, None
        type_id = self.area_type_id.get(area_id)
        return (self.area_name.get(area_id),
                self.area_type_name.get(type_id),
                self.country.get(area_id))


# ---------------------------------------------------------------------------
# Conexao
# ---------------------------------------------------------------------------

def connect(config_path: Path, work_mem: str, similarity_threshold: float,
            create_indexes: bool):
    """Abre a conexao, garante os indices e deixa o cursor pronto para o loop."""
    with open(config_path) as f:
        db = yaml.safe_load(f)["database"]

    conn = psycopg2.connect(
        host=db["ip"], database=db["name"], user=db["username"],
        password=db["password"], port=db["port"],
    )
    cur = conn.cursor()

    if create_indexes:
        # Sem estes indices cada busca faz seq scan em ~30M recordings. Na
        # primeira execucao a criacao demora bastante; depois é no-op.
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_artist_name_trgm
            ON musicbrainz.artist USING gin (name gin_trgm_ops);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_recording_name_trgm
            ON musicbrainz.recording USING gin (name gin_trgm_ops);
        """)
        # As buscas de release/release_group partem de `track.recording`. O dump
        # padrao ja traz `track_idx_recording`, entao isto normalmente é no-op --
        # fica como rede de seguranca para dumps parciais.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_track_recording
            ON musicbrainz.track (recording);
        """)
        conn.commit()

    # Loop so de leitura: autocommit evita segurar uma transacao aberta por horas.
    conn.autocommit = True
    cur.execute(f"SET work_mem = '{work_mem}';")
    cur.execute("SELECT set_limit(%s);", (similarity_threshold,))
    return conn, cur


class QueryRunner:
    """Executa a query com um cache LRU limitado.

    Guardar todos os pares unicos custaria dezenas de GB. Os pares repetidos ficam
    agrupados no dataset, entao um cache pequeno ja pega boa parte dos hits.

    Sao ~1,46M buscas no dataset inteiro: 611.537 relacoes x (1,227 + 1,165) artistas
    creditados em media nos dois lados. Buscar todos os creditos, e nao so o primeiro,
    tambem rende acerto de cache novo -- é comum o mesmo nome aparecer nos dois lados
    da mesma relacao (remix do proprio artista, por exemplo).
    """

    def __init__(self, cur, max_results, max_releases, cache_size):
        self.cur = cur
        self.max_results = max_results
        self.max_releases = max_releases
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __call__(self, artist_name, music_name):
        key = (artist_name, music_name)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        self.cur.execute(QUERY_COMPLEX, {
            "artist": artist_name, "track": music_name,
            "max_results": self.max_results, "max_releases": self.max_releases,
        })
        rows = self.cur.fetchall()

        if self.cache_size > 0:
            self.cache[key] = rows
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return rows


# ---------------------------------------------------------------------------
# Processamento dos resultados
# ---------------------------------------------------------------------------

def calculate_similarity(str1: str, str2: str) -> float:
    """Compara duas strings considerando suas versoes lower case sem espaco extra."""
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio()


def extract_musicbrainz_metadata(db_row, area_resolver=None) -> dict:
    """Extrai so o que queremos da linha crua, pra tirar lixo e reduzir espaco.

    Devolve um dict (e nao uma tupla) porque com release/release_group sao 11
    campos -- desempacotar por posicao a essa altura é convite pra erro silencioso
    quando alguem mexe na ordem.

    Os campos do artista (type, gender, area, begin_area, end_area) saem todos da
    propria linha `row_to_json(a)`; o `area_resolver` traduz os ids em nome e sobe a
    hierarquia ate o pais. Sem resolver, os campos derivados saem None -- serve pra
    reprocessar um lote ja gravado sem abrir conexao.
    """
    artist = db_row[0]
    recording = db_row[2]

    tag_recording_data = db_row[3]
    tag_artist_data = db_row[4]
    release_date = db_row[5]
    tag_release_data = db_row[6]
    tag_release_group_data = db_row[7]
    release_data = db_row[8]
    release_group_data = db_row[9]

    def tag_names(rows):
        return [t.get("name") for t in rows if "name" in t] if rows else []

    def artist_field(name):
        return artist.get(name) if artist else None

    if area_resolver is not None:
        area_name, area_type, country = area_resolver.area(artist_field("area"))
        begin_name, begin_type, begin_country = area_resolver.area(artist_field("begin_area"))
        end_name, end_type, end_country = area_resolver.area(artist_field("end_area"))
        artist_type = area_resolver.artist_type.get(artist_field("type"))
        gender = area_resolver.gender.get(artist_field("gender"))
    else:
        area_name = area_type = country = None
        begin_name = begin_type = begin_country = None
        end_name = end_type = end_country = None
        artist_type = gender = None

    # Identidade dos releases / release_groups, pra poder voltar ao MusicBrainz
    # depois sem refazer a busca fuzzy.
    releases = release_data or []
    release_groups = release_group_data or []

    return {
        "artist_gid": artist.get("gid") if artist else None,
        "recording_gid": recording.get("gid") if recording else None,
        "artist_tags": tag_names(tag_artist_data),
        "music_tags": tag_names(tag_recording_data),
        # Tags dos releases / release_groups, ja deduplicadas no SQL
        "release_tags": tag_names(tag_release_data),
        "release_group_tags": tag_names(tag_release_group_data),
        "releases": releases,
        "release_groups": release_groups,
        # Tipos primarios distintos (Album / Single / EP / Compilation ...), util
        # pra separar original de coletanea sem entrar na lista aninhada.
        "release_group_types": sorted({
            rg.get("primary_type") for rg in release_groups if rg.get("primary_type")
        }),
        # Metadados do artista, direto da linha de `artist`
        "artist_type": artist_type,      # Person / Group / Orchestra / ...
        "artist_gender": gender,         # Male / Female / Non-binary / ...
        # Area do artista: o nome cru pode ser de qualquer nivel (cidade, distrito,
        # estado...), entao vai junto o tipo e o pais resolvido pela hierarquia.
        "artist_area": area_name,
        "artist_area_type": area_type,
        "artist_country": country,
        # Onde comecou (nascimento / fundacao) e onde terminou -- mesma resolucao
        "artist_begin_area": begin_name,
        "artist_begin_area_type": begin_type,
        "artist_begin_country": begin_country,
        "artist_end_area": end_name,
        "artist_end_area_type": end_type,
        "artist_end_country": end_country,
        # Ano em que a musica foi lancada
        "release_year": release_date[0].get("year") if release_date else None,
    }


def process_and_validate_results(query_artist: str, query_song: str,
                                 db_matches: list, threshold: float,
                                 area_resolver=None) -> list:
    """Segundo filtro, em Python: o Postgres da recall (trigram), o difflib da precisao."""
    valid_results = []

    for row in db_matches:
        db_artist = row[0].get("name", "")
        db_song = row[2].get("name", "")

        artist_sim = calculate_similarity(query_artist, db_artist)
        song_sim = calculate_similarity(query_song, db_song)

        # Limita pelo threshold da biblioteca de diferencas
        if artist_sim > threshold and song_sim > threshold:
            valid_results.append({
                # qual dos nomes creditados produziu este match -- com varios
                # artistas por faixa, artist_similarity sozinho seria ambiguo
                "query_artist": query_artist,
                "database_artist": db_artist,
                "database_recording": db_song,
                "artist_similarity": artist_sim,
                "song_similarity": song_sim,
                # media harmonica: penaliza match bom num campo e ruim no outro
                "combined_similarity": (2 * (artist_sim * song_sim) / (artist_sim + song_sim)
                                        if (artist_sim + song_sim) > 0 else 0),
                **extract_musicbrainz_metadata(row, area_resolver),
                "pg_trgm_score": row[10],  # o score do Postgres
            })

    valid_results.sort(key=lambda x: x["combined_similarity"], reverse=True)
    return valid_results


def unique_names(artist_names) -> list:
    """Nomes nao vazios, sem repetidos (por caixa/espaco), na ordem dos creditos."""
    seen = set()
    names = []
    for name in artist_names or []:
        if not name or not name.strip():
            continue
        key = name.strip().lower()
        if key not in seen:
            seen.add(key)
            names.append(name)
    return names


def search_track(run_query, artist_names, music_name, threshold,
                 area_resolver=None, max_results=None):
    """Busca a faixa sob CADA artista creditado e junta o melhor de todos.

    O dataset credita mais de um artista em ~1 de cada 6 faixas (media 1,23 nomes em
    `by`, ate 7), e buscar so o primeiro perdia os demais: em "Smells Like"
    (Jarren Benton / R.A. The Rugged Man / Mic Buddah) so o primeiro nome achava 2
    matches, os tres juntos acham 6.

    Devolve (nomes buscados, matches). A dedup é por (artist_gid, recording_gid): a
    mesma gravacao PODE aparecer duas vezes se casar por artistas diferentes do
    MusicBrainz -- cada linha carrega o pais/tipo/tags do seu proprio artista, que é
    o que a analise usa. Repeticao exata fica com o de maior similaridade.
    """
    names = unique_names(artist_names)
    merged = {}

    for name in names:
        rows = run_query(name, music_name)
        for match in process_and_validate_results(name, music_name, rows, threshold,
                                                  area_resolver):
            key = (match["artist_gid"], match["recording_gid"])
            if key[0] is None or key[1] is None:  # gid sempre vem, mas nao custa
                key = (match["database_artist"], match["database_recording"])
            best = merged.get(key)
            if best is None or match["combined_similarity"] > best["combined_similarity"]:
                merged[key] = match

    # O LIMIT do SQL é por artista; o corte abaixo é o do conjunto -- sem ele uma
    # faixa de 7 creditos devolveria ate 350 resultados. Empates (combined 1.0 é
    # comum) ficam na ordem dos creditos, porque o sort do Python é estavel.
    matches = sorted(merged.values(), key=lambda m: m["combined_similarity"], reverse=True)
    if max_results is not None:
        matches = matches[:max_results]
    return names, matches


# ---------------------------------------------------------------------------
# Leitura dos lotes gravados
# ---------------------------------------------------------------------------

def iter_results(directory):
    """Percorre todos os lotes sem carregar nada grande na memoria."""
    directory = Path(directory)
    for path in sorted(directory.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                yield record["key"], record["relations"]


def load_batch(batch_no, directory):
    """Devolve um lote no formato do dict antigo ({key: {identifier_key: {...}}})."""
    path = Path(directory) / f"batch_{batch_no:05d}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {r["key"]: r["relations"] for r in map(json.loads, f)}


# ---------------------------------------------------------------------------
# Modos
# ---------------------------------------------------------------------------

def run_probe(args, run_query, area_resolver):
    """Consulta uma faixa so e imprime o JSON -- pra conferir a query sem rodar tudo."""
    *artists, track = args.probe
    t0 = time.perf_counter()
    searched, matches = search_track(run_query, artists, track, args.threshold,
                                     area_resolver, args.max_results)
    dt = time.perf_counter() - t0

    print(f"busca: {searched} / {track!r}", file=sys.stderr)
    print(f"{len(searched)} artista(s) consultado(s) em {dt:.2f}s -> "
          f"{len(matches)} acima do threshold {args.threshold}", file=sys.stderr)
    json.dump(matches[:args.probe_limit], sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


def run_stats(args):
    """Resume o que ja foi gravado: lotes, relacoes, taxa de match e cobertura de tags."""
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"diretorio nao existe: {output_dir}", file=sys.stderr)
        return 1

    done = len(list(output_dir.glob("*.done")))
    batches = len(list(output_dir.glob("*.jsonl.gz")))
    size = sum(p.stat().st_size for p in output_dir.glob("*.jsonl.gz"))
    print(f"lotes concluidos : {done}")
    print(f"arquivos .jsonl.gz: {batches}")
    print(f"tamanho em disco  : {size / 1e6:.1f} MB")

    if args.no_scan:
        return 0

    n_rel = n_side = n_match = n_multi = 0
    n_rec_tags = n_new_tags = n_no_tags = n_year = n_results = n_country = 0
    for _key, relations in tqdm(iter_results(output_dir), desc="lendo", unit=" rel",
                                disable=args.no_progress):
        n_rel += 1
        for rel in relations.values():
            for side in ("from", "to"):
                n_side += 1
                # nos lotes antigos artist_searched é string (um artista so)
                searched = rel[side].get("artist_searched")
                n_multi += isinstance(searched, list) and len(searched) > 1
                results = rel[side]["database_results"]
                n_match += bool(results)
                for m in results:
                    n_results += 1
                    n_year += m.get("release_year") is not None
                    n_country += m.get("artist_country") is not None
                    if m.get("music_tags"):
                        n_rec_tags += 1
                    elif m.get("release_tags") or m.get("release_group_tags"):
                        n_new_tags += 1
                    else:
                        n_no_tags += 1

    def pct(n, d):
        return f"{100 * n / d:5.1f}%" if d else "    -"

    print(f"\nrelacoes lidas    : {n_rel:,}")
    print(f"lados buscados    : {n_side:,}")
    print(f"  com 2+ artistas : {n_multi:,} ({pct(n_multi, n_side)})")
    print(f"  com match       : {n_match:,} ({pct(n_match, n_side)})")
    print(f"matches gravados  : {n_results:,}")
    print(f"  com release_year: {n_year:,} ({pct(n_year, n_results)})")
    print(f"  com pais        : {n_country:,} ({pct(n_country, n_results)})")
    print(f"  com music_tags  : {n_rec_tags:,} ({pct(n_rec_tags, n_results)})")
    print(f"  so release/rg   : {n_new_tags:,} ({pct(n_new_tags, n_results)})")
    print(f"  sem tag nenhuma : {n_no_tags:,} ({pct(n_no_tags, n_results)})")
    return 0


def run_mining(args, run_query, area_resolver):
    """Loop principal: streaming, em lotes e retomavel.

    O resultado completo nao cabe na RAM nem confortavelmente no disco, entao cada
    relacao é serializada e escrita na hora -- o pico de memoria é um item + o
    cache -- e cada lote vira um .jsonl.gz separado (~12x menor).

    O marcador .done é gravado so depois que o arquivo do lote fecha, entao um
    lote interrompido nao conta como pronto e é refeito na proxima execucao.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"lendo {args.input} ...", file=sys.stderr)
    with open(args.input) as f:
        whosampled_data = json.load(f)

    items = list(whosampled_data.items())
    if args.max_items is not None:
        items = items[:args.max_items]
    del whosampled_data

    n_batches = (len(items) + args.batch_size - 1) // args.batch_size
    print(f"{len(items):,} relacoes em {n_batches:,} lotes de {args.batch_size}",
          file=sys.stderr)

    # Uma barra so, contando relacoes (nao lotes): uma barra por lote anda uma vez
    # a cada varias horas e nao da ETA nenhum do run inteiro. O lote atual aparece
    # no postfix. smoothing baixo estabiliza o ETA num run dessa duracao.
    progress = tqdm(total=len(items), desc="relacoes", unit=" rel", smoothing=0.01,
                    mininterval=0.5, disable=args.no_progress)

    n_errors = 0
    t0 = time.perf_counter()
    try:
        for batch_no in range(n_batches):
            batch_path = output_dir / f"batch_{batch_no:05d}.jsonl.gz"
            done_marker = Path(f"{batch_path}.done")
            batch = items[batch_no * args.batch_size:(batch_no + 1) * args.batch_size]

            if done_marker.exists():
                progress.update(len(batch))  # lote ja pronto, so avanca a barra
                continue

            progress.set_postfix_str(f"lote {batch_no + 1}/{n_batches}")

            with gzip.open(batch_path, "wt", encoding="utf-8") as f:
                for key, item in batch:
                    relations = {}

                    for identifier_key, entry in item.items():
                        # todos os artistas creditados, nao so o primeiro
                        artist_names = [a["name"] for a in entry["by"]]
                        music_name = entry["track"]["name"]
                        other_artist_names = [a["name"] for a in entry["other_track_by"]]
                        other_music_name = entry["other_track"]["name"]

                        try:
                            searched, clean = search_track(
                                run_query, artist_names, music_name, args.threshold,
                                area_resolver, args.max_results)
                            searched_other, clean_other = search_track(
                                run_query, other_artist_names, other_music_name,
                                args.threshold, area_resolver, args.max_results)
                        except Exception as e:
                            n_errors += 1
                            print(f"Erro em {key}/{identifier_key}: {e}", file=sys.stderr)
                            continue

                        relations[identifier_key] = {
                            "from": {
                                "artist_searched": searched,
                                "music_searched": music_name,
                                "match_found": bool(clean),
                                "database_results": clean,
                            },
                            "to": {
                                "artist_searched": searched_other,
                                "music_searched": other_music_name,
                                "match_found": bool(clean_other),
                                "database_results": clean_other,
                            },
                        }

                    f.write(json.dumps({"key": key, "relations": relations},
                                       ensure_ascii=False) + "\n")
                    progress.update(1)

            done_marker.touch()
    except KeyboardInterrupt:
        print("\ninterrompido -- o lote em andamento nao foi marcado como pronto "
              "e sera refeito. Rode o mesmo comando pra retomar.", file=sys.stderr)
        return 130
    finally:
        progress.close()
        dt = time.perf_counter() - t0
        total = run_query.hits + run_query.misses
        if total:
            print(f"\ntempo {dt / 60:.1f} min | cache {run_query.hits:,}/{total:,} hits "
                  f"({100 * run_query.hits / total:.1f}%)", file=sys.stderr)
        if n_errors:
            print(f"{n_errors} entradas com erro (puladas)", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def optional_int(value):
    """int, ou None pra 'none'/'all' -- usado onde LIMIT NULL significa sem limite."""
    if value is None or str(value).lower() in ("none", "all", "todos", ""):
        return None
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("deve ser positivo (ou 'none')")
    return ivalue


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = p.add_argument_group("entrada / saida")
    io.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="yaml com as credenciais do Postgres (default: %(default)s)")
    io.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="json do WhoSampled (default: %(default)s)")
    io.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                    help="onde gravar os lotes .jsonl.gz (default: %(default)s)")

    match = p.add_argument_group("busca e filtro")
    match.add_argument("--similarity-threshold", type=float, default=0.55,
                       help="corte do pg_trgm pro operador %% (0..1). Baixar aumenta "
                            "o recall e o tempo (default: %(default)s)")
    match.add_argument("--threshold", type=float, default=0.55,
                       help="corte do difflib no filtro em Python (default: %(default)s)")
    match.add_argument("--max-results", type=optional_int, default=50,
                       help="melhores matches por faixa, ou 'none'. É o LIMIT do SQL por "
                            "artista creditado E o corte final da uniao dos artistas "
                            "(default: %(default)s)")
    match.add_argument("--max-releases", type=optional_int, default=20,
                       help="identidades de release guardadas por match, ou 'none'. As "
                            "TAGS agregam todos os releases; isto corta so a lista de "
                            "gid/nome, que explode em faixas reeditadas (default: %(default)s)")

    perf = p.add_argument_group("desempenho")
    perf.add_argument("--batch-size", type=int, default=100,
                      help="relacoes por arquivo de lote (default: %(default)s)")
    perf.add_argument("--cache-size", type=int, default=2000,
                      help="entradas no cache LRU de queries (default: %(default)s)")
    perf.add_argument("--work-mem", default="256MB",
                      help="work_mem da sessao; evita bitmap scan lossy nos indices "
                           "trigram (default: %(default)s)")
    perf.add_argument("--max-items", type=optional_int, default=None,
                      help="roda so as N primeiras relacoes (default: todas)")
    perf.add_argument("--skip-indexes", action="store_true",
                      help="nao roda os CREATE INDEX IF NOT EXISTS na largada (use se o "
                           "usuario do banco nao tiver permissao de DDL)")

    modes = p.add_argument_group("modos alternativos")
    modes.add_argument("--probe", nargs="+", metavar="ARTISTA... MUSICA",
                       help="consulta uma faixa so e imprime o JSON, sem gravar nada. O "
                            "ULTIMO argumento é a musica; todos os anteriores sao artistas")
    modes.add_argument("--probe-limit", type=int, default=3,
                       help="quantos matches o --probe imprime (default: %(default)s)")
    modes.add_argument("--stats", action="store_true",
                       help="resume o que ja foi gravado no --output-dir e sai")
    modes.add_argument("--no-scan", action="store_true",
                       help="com --stats, so conta os arquivos (nao le o conteudo)")

    p.add_argument("--no-progress", action="store_true", help="desliga a barra do tqdm")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.stats:
        return run_stats(args)

    # nargs="+" aceita um argumento so; precisa de pelo menos artista + musica
    if args.probe and len(args.probe) < 2:
        print("--probe precisa de pelo menos um artista e a musica, nessa ordem "
              "(ex: --probe \"Mobb Deep\" \"Shook Ones (Part 2)\")", file=sys.stderr)
        return 1

    # o --probe nao le o dataset, so o config
    required = [args.config] if args.probe else [args.config, args.input]
    for path in required:
        if not path.exists():
            print(f"arquivo nao encontrado: {path}", file=sys.stderr)
            return 1

    try:
        conn, cur = connect(args.config, args.work_mem, args.similarity_threshold,
                            create_indexes=not args.skip_indexes)
    except Exception as e:
        print(f"Erro ao conectar: {e}", file=sys.stderr)
        return 1

    run_query = QueryRunner(cur, args.max_results, args.max_releases, args.cache_size)
    try:
        # Pre-carrega a hierarquia de areas + tabelas de tipo/genero (~1s, uma vez)
        area_resolver = AreaResolver(cur, quiet=args.no_progress)
    except Exception as e:
        print(f"Erro ao carregar a hierarquia de areas: {e}", file=sys.stderr)
        cur.close()
        conn.close()
        return 1

    try:
        if args.probe:
            return run_probe(args, run_query, area_resolver)
        return run_mining(args, run_query, area_resolver)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
