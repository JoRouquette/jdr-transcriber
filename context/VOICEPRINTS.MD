# Empreintes vocales des joueurs

Ce guide explique comment enregistrer des **empreintes vocales** claires pour reconnaître automatiquement qui parle dans les transcriptions. Il est pensé pour être ajouté tel quel à ton dépôt.

## Objectif

Fournir à chaque joueur un court enregistrement (≈30 s) propre et naturel, utilisé par le pipeline pour mapper automatiquement `SPK1`, `SPK2`, etc. vers leurs **noms réels**.

## Où placer les fichiers

Mettre chaque enregistrement dans :
`context/voices/`

Exemples de noms de fichiers (un par joueur suffit) :

```
context/voices/Lindsay_scriptA.wav
context/voices/Margrim_scriptB.mp3
context/voices/Joueur3_scriptC.wav
```

Formats acceptés : `.wav` ou `.mp3`. Le pipeline convertit ensuite en **mono 16 kHz**.

## Réglages de prise de son (vite fait, mais bien)

- Lieu calme, sans musique.
- Micro à ~15 cm, légèrement de côté (évite les plosives “p/b”).
- Voix **naturelle**, pas de roleplay, pas de chuchotement.
- Débit normal, stable. Si un bruit ou un bafouillage survient, **recommencer**.

## Scripts à lire (≈30 secondes chacun)

Chaque joueur lit **l’un** des scripts ci-dessous (peu importe lequel), et enregistre un seul fichier propre.

### Script A — Français (couverture phonétique)

> « Je parle distinctement pour l’empreinte vocale. Ce test inclut voyelles claires et nasales : an, en, in, on, un. J’articule le r, le ch, le j, le gn, et les liaisons. Cinq, quinze, vingt; lundi, jeudi; janvier, juin; œil, écureuil, oignon. Je lis à vitesse normale, sans exagérer. Je compte de deux à douze, puis quatorze et dix-neuf. Je prononce aussi des mots fréquents : lumière, étrange, agile, tranquille, revenir. Ma voix reste stable, sans rire ni murmure. J’achève cette phrase en respirant calmement, comme pendant une conversation posée. Voilà. »

### Script B — Code-switch FR/EN (vocabulaire technique)

> « Je réalise une empreinte vocale naturelle. Parfois, à la table, on passe en anglais pour quelques mots. Je lis calmement. Today I’ll mention common terms: pull request, debug, browser, server, cache, schedule. Je dis aussi HTTP, API, JSON, et README. I keep a steady pace, no rush. Je prononce login, logout, commit, rollback, feature, hotfix. Je termine avec des chiffres et dates courtes : 2025, 13 juin, version 1.4.2. Je parle comme d’habitude, sans accent forcé ni voix jouée, pour que l’empreinte me reconnaisse facilement la prochaine fois. That’s all. »

### Script C — Narratif neutre (noms propres utiles au JdR)

> « Je lis un court passage neutre. Nous étions à Belienne, près d’Astralys, quand la rumeur d’Astraea a circulé. Dans l’auberge, la lumière vibrait comme un souvenir. J’entends encore le pas tranquille de Margrim, la voix de Lindsay, et les pages qu’on tourne. J’énonce des noms pour stabiliser l’orthographe : Aegasos, Abazil, Gueule de l’Orage, Athamé d’Agonie. Je garde un ton posé, ni théâtral, ni monotone. J’ajoute quelques mots précis : révélation, harmonie, silhouette, vallée, crépuscule. Je conclus sereinement, en gardant le même timbre, pour que l’empreinte soit nette et réutilisable. »

## Déclarer les empreintes dans le contexte YAML

Ouvre ou crée `context/context.yaml` et ajoute (ou complète) la section `speakers` avec le chemin de l’extrait pour chaque joueur :

```yaml
speakers:
  - id: 'Lindsay'
    role: 'PJ'
    nicknames: ['Linds']
    voice_samples:
      - 'voices/Lindsay_scriptA.wav'

  - id: 'Margrim'
    role: 'PJ'
    nicknames: []
    voice_samples:
      - 'voices/Margrim_scriptB.mp3'
```

Tu peux fournir **plusieurs** extraits par personne (le pipeline fera une moyenne des empreintes), mais **un seul bon** enregistrement suffit.

## Vérifier que tout est pris en compte

1. Les fichiers d’empreintes sont sous `context/voices/…`
2. `context/context.yaml` liste bien ces chemins dans `speakers[].voice_samples`.
3. Tu pushes sur la branche prévue (ex. `group/<quelquechose>`).
4. À la fin du run, ouvre l’artifact `transcripts` et vérifie que les segments portent les **noms** au lieu de `SPK1/SPK2`.

Astuce : si un nom reste incertain, tu peux forcer temporairement dans le YAML :

```yaml
overrides:
  speakers:
    SPK1: 'Lindsay'
```

## Dépannage rapide

- Son trop faible / saturé : refaire la prise, garder la même distance micro (~15 cm), éviter de parler “dans” le micro, viser un **angle léger**.
- Bruit de fond constant (ventilo, route) : changer de pièce ou couper la source si possible.
- Code-switch fréquent FR/EN : pas de souci, les scripts B/C et le pipeline en **multilingue** gèrent très bien.
