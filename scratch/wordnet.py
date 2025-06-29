from nltk.corpus import wordnet as wn


def forms2synsets():
    forms = {}
    for synset in wn.all_synsets():
        for form in synset.lemma_names():
            if form not in forms:
                forms[form] = set()
            forms[form].add(synset.name())
    return forms


def synset2definition(synset_name):
    synset = wn.synset(synset_name)
    return synset.definition()


forms = forms2synsets()

ss_apple = forms["apple"]

for apple in ss_apple:
    print(f"{apple}: {synset2definition(apple)}")
