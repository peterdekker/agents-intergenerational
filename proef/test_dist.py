import panphon.distance
import panphon.sonority
import numpy as np

np.set_printoptions(precision=3)

dst = panphon.distance.Distance()
son = panphon.sonority.Sonority()

def sonority_distance(a,b):
    return np.abs(son.sonority(a) - son.sonority(b))

def evaluate_dist(verb_type, distance_func, form, affix):
    form_border_phoneme = 0 if verb_type == "prefixing" else -1
    affix_border_phoneme = -1 if verb_type == "prefixing" else 0
    affix_slice = affix[affix_border_phoneme] if len(affix) > 0 else affix
    dist = distance_func(form[form_border_phoneme], affix_slice)
    return dist


verb_forms = {"suffixing": ["balik", "bu'a", "buka", "bəsuk", "de'in", "deka", "gehi", "gelu", "(gə)redo", "gəta", "haga", "həbo", "hitun", "hode", "horon", "kantar", "kirin", "kədoko", "kərian", "koda", "ləba wəkin", "lodo", "louk", "mia", "mori", "nyanyi", "ola", "peko", "peun", "pəla'e", "pupu", "səga", "taku", "tanin", "tei", "tobo", "tor", "turu", "tutu"],
              "prefixing": ["an", "a'an", "əte", "a'i", "enun", "oi", "ala", "ələ", "anan", "ahu' wai", "awa", "awan", "ian", "əwan", "itə", "iu", "odi", "olin", "urən"]}

affixes = {"prefixing": ["k", "m", "n", "m", "t", "m", "r"],
           "suffixing": ["kən", "ko", "no", "na","kən", "te", "ke", "ne", "ka"]}


distance_funcs = [("Levenshtein", dst.fast_levenshtein_distance),
                  ("Dogol prime distance", dst.dogol_prime_distance),
                  ("Feature edit distance", dst.feature_edit_distance),
                  ("Hamming feature edit distance", dst.hamming_feature_edit_distance),
                  ("Weighted feature edit distance", dst.weighted_feature_edit_distance),
                  ("Sonority distance", sonority_distance)
                 ]

for label, func in distance_funcs:
    print(f"---{label}---")
    for vt in ["prefixing", "suffixing"]:
        dists_dict = {f"{form}-{affix}": evaluate_dist(vt, func, form, affix) for form in verb_forms[vt] for affix in affixes[vt]}
        dists = list(dists_dict.values())
        mean = np.mean(dists)
        std = np.std(dists)
        minv = np.min(dists)
        maxv = np.max(dists)
        print(f"{vt}: mean {mean:.3f}, std {std:.3f}, min {minv:.3f}, max {maxv:.3f}")
        #print(dists_dict)
