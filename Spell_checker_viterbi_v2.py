import re
from collections import Counter
import operator
import nltk
import itertools

dict_list_onegram = open('big_t_En.txt').read()
dict_list_onegram = re.findall(r'\w+', dict_list_onegram.lower()) # generate single words from corpus

sent_bigrams = list(nltk.bigrams([w.lower() for w in dict_list_onegram]))
dict_list_bigram = [None]*len(sent_bigrams)
for index in range(len(sent_bigrams)):
    dict_list_bigram[index] = " ".join(sent_bigrams[index])

dict_list = dict_list_onegram + dict_list_bigram
WORDS = Counter(dict_list)

def candidates(word):
    "Generate possible spelling corrections for work"
    return (known([word]) or known(edits1(word)) or [word])
    #return (known([word]) | known(edits1(word))|set([word]))


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

    letters  = dict()
    replaces = list()
    for dd in word:
       letters[dd] = ''.join(prox_letter(dd))
    for cc in range(len(word)):
        for kk in letters[word[cc]]:
            replaces_temp = word[:cc] + kk + word[cc + 1:]
            replaces.append(replaces_temp)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(replaces + inserts + transposes + deletes)
def prox_letter(letr):
    array_prox = dict();
    array_prox['a'] = ['q', 'w', 'z', 'x','e','i'];
    array_prox['b'] = ['v', 'f', 'g', 'h', 'n'];
    array_prox['c'] = ['x', 's', 'd', 'f', 'v','k'];
    array_prox['d'] = ['x', 's', 'w', 'e', 'r', 'f', 'v', 'c'];
    array_prox['e'] = ['w', 's', 'd', 'f', 'r','a'];
    array_prox['f'] = ['c', 'd', 'e', 'r', 't', 'g', 'b', 'v'];
    array_prox['g'] = ['r', 'f', 'v', 't', 'b', 'y', 'h', 'n'];
    array_prox['h'] = ['b', 'g', 't', 'y', 'u', 'j', 'm', 'n'];
    array_prox['i'] = ['u', 'j', 'k', 'l', 'o','a'];
    array_prox['j'] = ['n', 'h', 'y', 'u', 'i', 'k', 'm'];
    array_prox['k'] = ['u', 'j', 'm', 'l', 'o','c'];
    array_prox['l'] = ['p', 'o', 'i', 'k', 'm'];
    array_prox['m'] = ['n', 'h', 'j', 'k', 'l'];
    array_prox['n'] = ['b', 'g', 'h', 'j', 'm'];
    array_prox['o'] = ['i', 'k', 'l', 'p','a','e'];
    array_prox['p'] = ['o', 'l'];
    array_prox['q'] = ['w', 'a', 'c'];
    array_prox['r'] = ['e', 'd', 'f', 'g', 't'];
    array_prox['s'] = ['q', 'w', 'e', 'd', 'c', 'x', 'c', 'x', 'z', 'a'];
    array_prox['t'] = ['r', 'f', 'g', 'h', 'y','d'];
    array_prox['u'] = ['y', 'h', 'j', 'k', 'i','o'];
    array_prox['v'] = [' ', 'c', 'd', 'f', 'g', 'b'];
    array_prox['w'] = ['q', 'a', 's', 'd', 'e'];
    array_prox['x'] = ['z', 'a', 's', 'd', 'c'];
    array_prox['y'] = ['t', 'g', 'h', 'j', 'u','e','s'];
    array_prox['z'] = ['x', 's', 'a'];
    array_prox['1'] = ['q', 'w'];
    array_prox['2'] = ['q', 'w', 'e'];
    array_prox['3'] = ['w', 'e', 'r'];
    array_prox['4'] = ['e', 'r', 't'];
    array_prox['5'] = ['r', 't', 'y'];
    array_prox['6'] = ['t', 'y', 'u'];
    array_prox['7'] = ['y', 'u', 'i'];
    array_prox['8'] = ['u', 'i', 'o'];
    array_prox['9'] = ['i', 'o', 'p'];
    array_prox['0'] = ['o', 'p'];
    array_prox[' '] = ['x','c','v','b','n','m']

    return ''.join(array_prox[letr])

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correction_viterbi(query):
    word_list = re.findall(r'\w+', query.lower())
    nn = len(word_list)
    if nn == 1:
        corr_list = [None]
        word_list_corrected_temp = (candidates(word_list[0]))
        prob_list = dict()
        for index in word_list_corrected_temp:
            prob_list[index] = WORDS[index]
        word_list_corrected = max(prob_list.iteritems(), key=operator.itemgetter(1))[0]
        corr_list[0] = word_list_corrected
        corrected_str = " ".join(corr_list)
    else:
        i = 0
        cand_list = dict()
        while i < nn:
            word_list_corrected_temp = list(candidates(word_list[i]))
            cand_list[i] = word_list_corrected_temp
            i+=1

        sent_plus_score = dict()
        for t in itertools.product(*cand_list.values()):
            bigram_temp_prob = 0
            for index in range(len(t)-1):
                bigram_temp = " ".join(t[index:index+2])
                bigram_temp_prob = WORDS[bigram_temp] + WORDS[t[index]] + WORDS[t[index+1]] + bigram_temp_prob
                #print (bigram_temp,WORDS[bigram_temp])
            sent_plus_score[t] = bigram_temp_prob
            #print sent_plus_score
        word_list_corrected = max(sent_plus_score.iteritems(), key=operator.itemgetter(1))[0]
        corrected_str = " ".join(word_list_corrected)
        print corrected_str
    return corrected_str

def correction(query):
    word_list = re.findall(r'\w+', query.lower())
    nn = len(word_list)
    if nn == 1:
        corr_list = [None]
        #print (candidates(word_list[0]))
        word_list_corrected_temp = (candidates(word_list[0]))
        prob_list = dict()
        for index in word_list_corrected_temp:
            prob_list[index] = WORDS[index]
        word_list_corrected = max(prob_list.iteritems(), key=operator.itemgetter(1))[0]
        corr_list[0] = word_list_corrected
        corrected_str = " ".join(corr_list)
    else:
        if nn%2 == 0:
            corr_list = [None]
            i = 0
            while i < nn/2:
                if i == (nn-2)/2:
                    #print 'one to last word!'
                    word_list_corrected_temp = list(candidates(word_list[2*i] + " " + word_list[2*i+1]))
                    #print word_list_corrected_temp
                    prob_list = dict()
                    for index in word_list_corrected_temp:
                        prob_list[index] = WORDS[index]
                    word_list_corrected = max(prob_list.iteritems(), key=operator.itemgetter(1))[0]
                    if i == 0:
                        corr_list[0] = word_list_corrected
                    else:
                        corr_list.append(word_list_corrected)
                else:
                    #print 'rest of the words!'
                    word_list_corrected_temp = list(candidates(word_list[2*i] + " " + word_list[2*i+1]))
                    #print word_list_corrected_temp
                    prob_list = dict()
                    for index in word_list_corrected_temp:
                        prob_list[index] = WORDS[index]
                    word_list_corrected= max(prob_list.iteritems(), key=operator.itemgetter(1))[0]
                    if i == 0:
                        corr_list[0] = word_list_corrected
                    else:
                        corr_list.append(word_list_corrected)
                i += 1
        else:
            corr_list = [None]
            i = 0
            while i < (nn-1)/2:
                if i == (nn-3)/2:
                    #print 'one to last word!'
                    word_list_corrected_temp = list(candidates(word_list[2*i] + " " + word_list[2*i+1]))
                    #print word_list_corrected_temp
                    prob_list = dict()
                    for index in word_list_corrected_temp:
                        prob_list[index] = WORDS[index]
                    word_list_corrected = max(prob_list.iteritems(), key=operator.itemgetter(1))[0]
                    if i == 0:
                        corr_list[0] = word_list_corrected
                    else:
                        corr_list.append(word_list_corrected)
                else:
                    #print 'rest of the words!'
                    word_list_corrected_temp = list(candidates(word_list[2*i] + " " + word_list[2*i+1]))
                    #print word_list_corrected_temp
                    prob_list = dict()
                    for index in word_list_corrected_temp:
                        prob_list[index] = WORDS[index]
                    word_list_corrected= max(prob_list.iteritems(), key=operator.itemgetter(1))[0]
                    if i == 0:
                        corr_list[0] = word_list_corrected
                    else:
                        corr_list.append(word_list_corrected)
                i += 1
            #print 'last word!'
            word_list_corrected_temp = list(candidates(word_list[-2] + " " + word_list[-1]))
            #print word_list_corrected_temp
            prob_list = dict()
            for index in word_list_corrected_temp:
                prob_list[index] = WORDS[index]
            word_list_corrected = max(prob_list.iteritems(), key=operator.itemgetter(1))[0]
            corr_list.append(word_list_corrected.split()[1])
            #print word_list_corrected.split()[1]
        corrected_str = " ".join(corr_list)
    return corrected_str
