# Gold Standard Dataset for Baybayin Disambiguation

## Overview
This dataset contains manually created Filipino sentences for testing Baybayin disambiguation.
Each ambiguous pair has **50 sentences per sense** (meaning).

## Dataset Structure

```
gold_standard_dataset/
├── sentences/           # All sentence files by ambiguous pair
│   ├── 01_asero_asido.txt
│   ├── 02_bote_buti.txt      ✅ Complete
│   ├── 03_boto_buto.txt
│   └── ... (15 files total)
├── combined/
│   └── all_sentences.txt     # All combined for final dataset
├── tracking.md
└── README.md
```

## Ambiguous Pairs (16 total) - 1,550 sentences needed

| # | Pair | Senses | Sentences | Status |
|---|------|--------|-----------|--------|
| 1 | asero, asido | 2 | 100 | ⬜ |
| 2 | **bote, buti** | 2 | 100 | ✅ Complete |
| 3 | boto, buto | 2 | 100 | ⬜ |
| 4 | higante, higanti | 2 | 100 | ⬜ |
| 5 | hito, heto | 2 | 100 | ⬜ |
| 6 | itodo, ituro | 2 | 100 | ⬜ |
| 7 | kamada, kamara | 2 | 100 | ⬜ |
| 8 | kompas, kumpas | 2 | 100 | ⬜ |
| 9 | kumita, kometa | 2 | 100 | ⬜ |
| 10 | mesa, misa | 2 | 100 | ⬜ |
| 11 | polo, pulo | 2 | 100 | ⬜ |
| 12 | poso, puso | 2 | 100 | ⬜ |
| 13 | tela, tila | 2 | 100 | ⬜ |
| 14 | todo, toro, turo | 3 | 150 | ⬜ |
| 15 | toyo, tuyo | 2 | 100 | ⬜ |

## Word Meanings Reference

| Pair | Word 1 | Word 2 | Word 3 |
|------|--------|--------|--------|
| 1 | **asero** = steel | **asido** = acid | |
| 2 | **bote** = bottle | **buti** = goodness/fortunately | |
| 3 | **boto** = vote | **buto** = bone/seed | |
| 4 | **higante** = giant | **higanti** = revenge | |
| 5 | **hito** = catfish | **heto** = here it is | |
| 6 | **itodo** = go all out | **ituro** = point/teach | |
| 7 | **kamada** = stack/pile | **kamara** = camera/chamber | |
| 8 | **kompas** = compass | **kumpas** = wave/gesture | |
| 9 | **kumita** = to earn | **kometa** = comet | |
| 10 | **mesa** = table | **misa** = mass (religious) | |
| 11 | **polo** = polo shirt | **pulo** = island | |
| 12 | **poso** = well (water) | **puso** = heart | |
| 13 | **tela** = cloth/fabric | **tila** = seems like | |
| 14 | **todo** = full force | **toro** = bull | **turo** = teaching |
| 15 | **toyo** = soy sauce | **tuyo** = dried fish/dry | |

## Sentence Requirements

1. **Natural**: Each sentence should sound natural to a Filipino speaker
2. **Clear context**: The context should make the intended word unmistakable
3. **Varied**: Use different sentence structures and contexts
4. **Complete**: Each sentence should be grammatically complete

## Usage

Run the test script:
```bash
python test_bote_buti.py  # Test single pair
python evaluate.py        # Full evaluation
```
