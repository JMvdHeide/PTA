#!/usr/bin/env python3

import glob
import wikipedia
import warnings
from nltk.tag.stanford import StanfordNERTagger
from nltk.wsd import lesk


def transform_tag(tag, word, words):
    synset = lesk(words, word, "n")
    if synset:
        if tag == "ORGANIZATION" or tag == "PERSON":
            return tag[:3]
        elif tag == "LOCATION":
            paths = synset.hypernym_paths()
            for path in paths:
                for synset in path:
                    name = synset.name()
                    if "city" in name or "town" in name:
                        return "CIT"
                    elif "country" in name or "state" in name:
                        return "COU"
            return "NAT"
        elif tag == "MISC":
            paths = synset.hypernym_paths()
            for path in paths:
                for synset in path:
                    name = synset.name()
                    if "animal" in name:
                        return "ANI"
                    elif "sport" in name:
                        return "SPO"
                    elif "entertainment" in name:
                        return "ENT"
            return ""
        else:
            return ""
    else:
        return ""


def wiki_search(query):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wikipage = wikipedia.search(query)[0]
            wikiurl = wikipedia.page(wikipage).url
    except wikipedia.exceptions.DisambiguationError as e:
        wikiurl = wiki_disambiguate(e.options)

    return wikiurl


def wiki_disambiguate(options, n=0):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wikiurl = wikipedia.page(options[n]).url
    except wikipedia.exceptions.DisambiguationError:
        return wiki_disambiguate(options, n+1)

    return wikiurl


def main():
    pathname = "test/*/*/*.pos"
    directory = glob.glob(pathname)
    for f in directory:
        with open(f) as readfile:
            print(f)
            print("Processing...")
            result = []
            history = []
            chunk = False

            # Collect words
            words = [line.split()[3] for line in readfile if len(line.split()) > 1]

            # NER tag using Stanford
            stanford = StanfordNERTagger('stanford-ner-2014-06-16/classifiers/english.conll.4class.distsim.crf.ser.gz', 'stanford-ner-2014-06-16/stanford-ner-3.4.jar')
            tagged = stanford.tag(words)

            for word_tuple in tagged:
                word = word_tuple[0]
                tag = word_tuple[1]
                # Determine new tag
                new_tag = transform_tag(tag, word, words)

                # Determine chunk
                if len(history) > 0 and history[-1][1] != new_tag:
                    if len(history) > 1 and history[-1][1]:
                        chunk = " ".join([tpl[0] for tpl in history])
                    history = []
                new_word_tuple = (word, new_tag)
                history.append(new_word_tuple)

                # Search wikipedia page
                # Process chunk
                if chunk:
                    wikiurl = wiki_search(chunk)

                    chunk_length = len(chunk.split())
                    old_combis = result[-chunk_length:]
                    result = result[:-chunk_length]
                    for old_combi in old_combis:
                        new_combi = old_combi[:-1]
                        new_combi.append(wikiurl)
                        result.append(new_combi)

                    chunk = False

                # Process word
                if new_tag:
                    wikiurl = wiki_search(word)
                else:
                    wikiurl = ""

                result.append([word, new_tag, wikiurl])

        # Write results to .ent.aut file
        with open(f) as readfile2, open(f + ".ent.aut", "a") as writefile:
            print("Writing...")
            n = 0
            for line in readfile2:
                if len(line) > 1:
                    new_line = line.rstrip() + " " + result[n][1] + " " + result[n][2]
                    print(new_line, file=writefile)
                n += 1

if __name__ == "__main__":
    main()
