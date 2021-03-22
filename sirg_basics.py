from rebuild_js import author_to_gaffiliations#, author_to_gh_index
from grab_js import coords_of_target_university#, university_data_frame
from typing import List
import requests
#from science_access.online_app_backend import author_to_urls, visit_link, unpaywall_semantic_links

from tqdm import tqdm

import requests
import pprint

import pickle
import os
from natsort import natsorted, ns
import semanticscholar as sch


def author_to_urls(NAME):
    '''
    splash_url is a URL where Dissemin thinks that the paper is described, without being necessarily available. This can be a publisher webpage (with the article available behind a paywall), a page about the paper without a copy of the full text (e.g., a HAL page like https://hal.archives-ouvertes.fr/hal-01664049), or a page from which the paper was discovered (e.g., the profile of a user on ORCID).
    pdf_url is a URL where Dissemin thinks the full text can be accessed for free. This is rarely a direct link to an actual PDF file, i.e., it is often a link to a landing page (e.g., https://arxiv.org/abs/1708.00363). It is set to null if we could not find a free source for this paper.
    '''
    response = requests.get("https://dissem.in/api/search/?authors=" + str(NAME))
    author_papers = response.json()
    visit_urls = []
    coauthors = []
    titles = []
    papers = []
    for p in tqdm(author_papers["papers"]):
        coauthors_ = p["authors"]
        title = p["title"]
        titles.append(title)
        coauthors.append(coauthors_)
        records = p["records"][0]
        if "doi" in records.keys():
            visit_urls.append(records["doi"])
            paper =  sch.paper(records["doi"], timeout=12)# for doi_ in tqdm(dois) ]
            #import pdb
            #pdb.set_trace()
            papers.append(paper)
    print(len(papers))

    dois = visit_urls
    print(dois)
    return dois, coauthors, titles, visit_urls, papers

def unpaywall_semantic_links(name,local_dict):
    """
    inputs a URL that's full of publication orientated links, preferably the
    authors scholar page.
    """
    dois, coauthors, titles, visit_urls,papers = author_to_urls(name)
    #assert len(papers)!=0
    #papers = [ sch.paper(doi_, timeout=12) for doi_ in tqdm(dois) ]

    for paper in tqdm(papers):
        if 'authors' in paper.keys():
            for author_ in paper['authors']:
                if author_['name'] not in local_dict.keys():
                    local_dict[author_['name']] = {}

                if 'citations' in paper.keys():
                    if 'paper_citations' not in local_dict[author_['name']].keys():
                        local_dict[author_['name']]['paper_citations'] = []
                    local_dict[author_['name']]['paper_citations'].append(len(paper['citations']))
                if 'isInfluential' in paper.keys():

                    local_dict[author_['name']]['isInfluential'] = paper['isInfluential']
                if 'citationVelocity' in paper.keys():
                    local_dict[author_['name']]['paper_citationVelocity'] = paper['citationVelocity']

                author = sch.author(author_['authorId'], timeout=12)

                if 'aliases' in author.keys():
                    local_dict[author_['name']]['aliases'] = author['aliases']
                if 'citationVelocity' in author.keys():
                    local_dict[author_['name']]['citationVelocity'] = author['citationVelocity']
                if 'influentialCitationCount' in author.keys():
                    local_dict[author_['name']]['influentialCitationCount'] = author['influentialCitationCount']

    print(len(local_dict),'local dict len')

    #if len(local_dict) == 0:
    #    local_dict[name] = {}
    return local_dict


sirg_author_list = [
    "Brian H. Smith",
    "Christian Rabeling",
    "Jon F. Harrison",
    "Juergen Liebig",
    "Stephen C. Pratt",
    "Jennifer H. Fewell",
    "Arianne J. Cease",
    "Gro V. Amdam"]

sirg_author_list2 = [
    "B. H. Smith",
    "C. Rabeling",
    "J. F. Harrison",
    "J. Liebig",
    "S. C. Pratt",
    "J. H. Fewell",
    "A. J. Cease",
    "G. Amdam"]

sirg_author_list3 = [
    "J. Harrison",
    "S. Pratt",
    "A. Cease"]

sirg_author_list.extend(sirg_author_list2)
sirg_author_list.extend(sirg_author_list3)

if False:
    #if os.path.exists('global_dict.p'):
    with open('global_dict.p','rb') as f:
        global_dict = pickle.load(f)
else:
    global_dict = {}

    for name in sirg_author_list:
        ##
        # put through the geolocating pipeline.
        ##
        local_dict = unpaywall_semantic_links(name,global_dict)
        global_dict.update(local_dict)
        with open('global_dict.p','wb') as f:
            pickle.dump(global_dict,f)
        print(len(global_dict),'global_dict len')
    #author_results, visit_urls = visit_link(visit_more_urls):
def hIndex(citations:List)->int:
    """
    From:
    https://gist.github.com/restrepo/c5f8f9fd5504a3f93ae34dd10a5dd6b0
    https://github.com/kamyu104/LeetCode/blob/master/Python/h-index.py
    """
    print(sorted(citations))
    return sum(x >= i + 1 for i, x in enumerate(sorted(  list(citations), reverse=True)))

Smith = hIndex(global_dict['B. Smith']['paper_citations'])
Rabeling = hIndex(global_dict['C. Rabeling']['paper_citations'])
Amdam = global_dict["G. Amdam"]['influentialCitationCount']
sirg = {}
sirg['Amdam'] = global_dict["G. Amdam"]['influentialCitationCount']
sirg['Rabeling'] = global_dict["C. Rabeling"]['influentialCitationCount']
sirg['Smith'] = global_dict["B. Smith"]['influentialCitationCount']
sirg['Fewell'] = global_dict["J. Fewell"]['influentialCitationCount']
ns = natsorted(global_dict.keys())
for n in ns:
    print(n)
