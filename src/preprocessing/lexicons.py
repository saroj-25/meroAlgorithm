"""Lexicons and regexes for Romanized-Nepali / English code-switch detection.

These lists are deliberately small and inspectable.  They are used for
(a) weak-labelling the token-level training data for the character n-gram
language-ID classifier (Section 3.3 of the paper) and (b) as a rule-based
fallback when no trained classifier checkpoint is available.

The three label classes follow the paper:
    nepali     - Romanized Nepali token
    english    - English token
    ambiguous  - script-ambiguous / technical token (``O(n log n)``, ``DFS``, ``i++``)
"""

from __future__ import annotations

import re
from typing import List

# --------------------------------------------------------------------------- #
# Romanized Nepali function words, question words and verb forms.
# Multiple spellings are listed on purpose: Romanized Nepali has no standard
# orthography (paper, Section 2.3), e.g. huncha / hunchha / hunxa.
# --------------------------------------------------------------------------- #
NEPALI_WORDS = set("""
k ke kya kasari kasto kastoo kati kina kahile kaha kahaa kun kunai kasle kasko
ko ka ki lai bata sanga sangai ma mai maa nai
yo tyo yi ti yesto tyasto yesari tyasari yaha tyaha
bhaneko bhanne bhanera bhanchan bhanchha bhancha bhanda bhandaa
ho hoina hola holaa haina huncha hunchha hunxa hudaina hune huney bhayo bhaeko bhaisakyo
cha chha xa chan chhan xan thiyo thiye
garne garnu garcha garchha garxa garda garepachi garisakepachi garnus gareko garera garna
lekhne lekhnu lekhda lekheko lekhnus
herne hernu herda hernus heryo
bujhne bujhnu bujhda bujhincha bujhna bujhaunu bujhayo bujhena
sikne siknu sikda sikaunu
aauncha aauchha aauxa aauna aayo aaudaina
milcha milchha milxa milyo milaunu milaune
parcha parchha parxa parne pardaina paryo
chahincha chahiyo chahine chahanchu
sakcha sakchha sakinccha sakdaina sakincha
dinu dincha dinchha diyo deko diyeko
rakhne rakhnu rakhnus rakheko
khojne khojnu khojda khojeko
tara ani ra pani matra matrai athawa athaba ki
dherai thorai ali ekdam ekchoti ekpalta palta choti
sajilo garho gaaro sahaj ramro naramro thik galat galti
samasya prashna prasna jawaf uttar sodhne sodhnu sodhda sodheko sodhna
pahila pachi pachhi bich bich-ma mathi tala agadi pachadi
sabai sab aru arko arka feri pheri
mero timro hamro usko unko aafno aaphno aafai aaphai
ma hami timi tapai tapaai u uni uniharu
kelai kehi kei kesari
sano thulo lamo chhoto chhotai naya purano
banaune banaunu banaucha banayo
chalcha chalchha chaldaina chalayo
lagcha lagchha lagyo laj
milaune nikalne nikalnu nikalda nikaleko
""".split())

# --------------------------------------------------------------------------- #
# Frequent English words in student queries (function words + course verbs).
# --------------------------------------------------------------------------- #
ENGLISH_WORDS = set("""
a an the is are was were be been am do does did done doing
what why how when where which who whose whom
this that these those it its there here
i you we they he she me my your our their his her
and or but not with without for from to of in on at by as into
can could should would will shall may might must
explain define describe compare contrast implement write give show tell prove derive
difference between example examples exercise question answer solution step steps
time space complexity worst best average case cases bound bounds
algorithm algorithms data structure structures code program function method
please help need want understand understanding confused doubt
work works working use uses used using apply applied
if else while loop loops recursion recursive iterative iteration
input output array arrays list lists node nodes pointer pointers value values key keys
sort sorted sorting search searching insert insertion delete deletion update
tree trees graph graphs stack stacks queue queues heap heaps hash hashing string strings
""".split())

# --------------------------------------------------------------------------- #
# Technical / script-ambiguous tokens.
# --------------------------------------------------------------------------- #
TECH_TOKENS = set("""
dfs bfs bst avl dsu mst scc lru lcs lis dp kmp rrf bm25 faiss hnsw api gpu cpu
o(n) o(1) o(logn) nlogn logn n^2 n2 2^n on
null nil none true false int char float double void return
i++ j++ ++i n-1 n+1 a[i] arr[i] dp[i] dp[i][j] left right mid lo hi
push pop peek enqueue dequeue heapify partition pivot memoization tabulation
""".split())

TECH_PATTERNS = [
    re.compile(r"^[oO]\(.*\)$"),          # O(n log n)
    re.compile(r"^[A-Z]{2,6}$"),          # DFS, BST, AVL
    re.compile(r"^\w+\[[^\]]*\]$"),       # arr[i], dp[i][j]
    re.compile(r"[+\-*/=<>^%]{1,2}$"),    # i++, n-1
    re.compile(r"^\d+$"),                 # 5, 100
    re.compile(r"^\w+\(\)$"),             # push(), pop()
]

TOKEN_RE = re.compile(r"[A-Za-z\u0900-\u097F]+(?:\[[^\]]*\])?|\S+")

LABELS = ["nepali", "english", "ambiguous"]


def tokenize(text: str) -> List[str]:
    """Whitespace-and-punctuation tokenizer that keeps technical tokens intact."""
    return [t for t in re.split(r"\s+", text.strip()) if t]


def clean_token(token: str) -> str:
    """Lowercase and strip trailing punctuation, keeping code-like characters."""
    return token.strip(" ,.;:!?\"'`\u2018\u2019\u201c\u201d").lower()


def is_technical(token: str) -> bool:
    raw = token.strip(" ,.;:!?\"'")
    low = clean_token(token)
    if low in TECH_TOKENS:
        return True
    return any(p.match(raw) for p in TECH_PATTERNS)


def rule_label(token: str) -> str:
    """Weak label for a single token: nepali | english | ambiguous."""
    if is_technical(token):
        return "ambiguous"
    low = clean_token(token)
    if not low:
        return "ambiguous"
    if low in NEPALI_WORDS:
        return "nepali"
    if low in ENGLISH_WORDS:
        return "english"
    # Orthographic heuristics for unseen words.
    if re.search(r"(chha|ncha|nxa|garn|bhan|hunc|aunc|nus$|eko$|dai$|lai$)", low):
        return "nepali"
    if re.search(r"(tion$|ment$|ing$|ness$|ity$|ical$|ise$|ize$)", low):
        return "english"
    return "ambiguous"


def rule_language_profile(text: str) -> dict:
    """Proportion of each class in a string (rule-based, no model needed)."""
    tokens = tokenize(text)
    if not tokens:
        return {"nepali": 0.0, "english": 0.0, "ambiguous": 1.0}
    labels = [rule_label(t) for t in tokens]
    return {lab: labels.count(lab) / len(labels) for lab in LABELS}
