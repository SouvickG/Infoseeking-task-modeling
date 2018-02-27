from __future__ import print_function
from __future__ import division
import xml.etree.ElementTree as ET
import pandas as pd
import copy
import numpy as np
import itertools
from nltk.corpus import stopwords
from stop_words import get_stop_words
import scipy.stats
from collections import Counter
import sys
import os
sys.path.append(os.path.abspath('../external/amur-query-reformulation'))
import reformulationClassifierFunctions as reformulations






class NSFLogParser:
    # Data frame with rows:
    # <session_num, user_num, topic_num, count interactions>

    # TODO: What are the statistics on session times?

    def __init__(self,output_csv='./output.csv'):
        self.output_csv = output_csv
        # self.topicqueries_to_users = dict()
        # self.topicurls_to_users = dict()
        # self.topicqueryterms_to_users = dict()
        self.questionwords = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
        self.conjunctionwords = ['and', 'or', 'nor', 'for', 'but', 'so', 'yet']
        self.linkingverbs = ['am', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'has', 'have', 'had', 'can', 'may',
                        'do', 'does', 'did', 'will', 'shall', 'might', 'must', 'could', 'would', 'should']
        self.prepositions = ['with', 'on', 'for', 'after', 'at', 'by', 'in', 'against', 'near', 'between', 'off', 'from',
                        'through', 'over', 'up']
        self.stopwords = set(stopwords.words('english') + get_stop_words('en'))

        tree = ET.parse('../../data/external/trec2014/sessiontrack2014.xml')
        self.root = tree.getroot()

        self.judgmentsbytopic = dict()
        with open('../../data/external/trec2014/judgments.txt') as judgmentsfile:
            for line in judgmentsfile:
                (topicID, ignore, docid, relevance) = line.split()
                (topicID, ignore, docid, relevance) = (int(topicID), ignore, docid, int(relevance))
                print((topicID, ignore, docid, relevance))
                if self.judgmentsbytopic.get(topicID) is None:
                    self.judgmentsbytopic[topicID] = dict()
                    self.judgmentsbytopic[topicID][docid] = relevance

        # tce_annotation_sheets = pd.read_excel('../../data/external/trec2014/2017-09-23_facetannotations_annotator1.xlsx',sheetname='search task annotation')
        tce_annotation_sheets = pd.read_csv('../../data/external/trec2014/sessiontopicmap14.tsv',
                                            sep='\t')

        tasktypes = []
        for (n, group) in tce_annotation_sheets.groupby(['topic']):
            tasktypes += [group['tasktype'].tolist()[0]]

        self.topic_to_product = dict(zip(tce_annotation_sheets['topic'], tce_annotation_sheets['product']))
        self.topic_to_goal = dict(zip(tce_annotation_sheets['topic'], tce_annotation_sheets['goal']))
        self.topic_to_trectasktype = dict(zip(tce_annotation_sheets['topic'], tce_annotation_sheets['tasktype']))

        VALID_TOPICS = [i for i in range(1, 61)]
        VALID_USERTOPICS = dict()
        userID_none = -1
        # Users and completed users
        df = []
        facets_eda_df = []
        overview_df = []
        query_length_means = []
        query_length_variances = []



    def _query_jaccard(self,q1, q2):
        q1 = set(q1.split())
        q2 = set(q2.split())
        return len(q1 & q2) / len(q1 | q2)

    def _jaccard_list(self,l1, l2):
        s1 = set(l1)
        s2 = set(l2)
        return len(s1 & s2) / len(s1 | s2)

    def _jaccard_querypairs(self,queries, lim=10):
        jaccards = []
        for i in range(len(queries) - 1):
            q1, q2 = queries[i], queries[i + 1]
            results1, results2 = q1[:lim], q2[:lim]
            # print("RES1\n",results1,"RES2\n",results2)
            results1_concat, results2_concat = ' '.join(results1), ' '.join(results2)
            results1_wordlist, results2_wordlist = results1_concat.split(' '), results2_concat.split(' ')
            jaccards += [self._jaccard_list(results1_wordlist, results2_wordlist)]
        return jaccards

    def _make_df_rows(self,curr_row, prev_row):

        # finish row: previous action was completed. None if just started action
        # start row: action that was started.  None if at end of sequence
        finish_df_row = {'userID': curr_row['userID'], 'questionID': curr_row['questionID'],
                         'facet_trec2014tasktype': curr_row['facet_trec2014tasktype'],
                         # 'facet_goalplusproduct': curr_row['facet_goal'] + curr_row['facet_product'],
                         'facet_goal': curr_row['facet_goal'],
                         'facet_product': curr_row['facet_product']
                         # 'facet_complexity': curr_row['facet_complexity'],
                         # 'facet_complexity_merged': curr_row['facet_complexity_merged'],
                         # 'facet_complexity_create': curr_row['facet_complexity_create']
                         }
        start_df_row = copy.deepcopy(finish_df_row)

        questionID = curr_row['questionID']

        # Get data for when a page/query action is finished
        if curr_row is not None:
            start_df_row['session_num'] = curr_row['session_num']
            start_df_row['action_type'] = curr_row['action_type']
            start_df_row['start_time'] = curr_row['start_time']
            start_df_row['interaction_num'] = curr_row['interaction_num']
            start_df_row['total_actions_count'] = curr_row['total_actions_count']

            start_df_row['pages_num_session'] = curr_row['pages_num_session']
            start_df_row['pages_numunique_session'] = len(set(curr_row['urls_session']))
            start_df_row['queries_num_session'] = curr_row['queries_num_session']
            start_df_row['serps_num_session'] = curr_row['serps_num_session']
            # # start_df_row['clicks_num_session'] = curr_row['clicks_num_session']
            #
            #
            start_df_row['perquery_pages'] = curr_row['pages_num_session'] / curr_row['queries_num_session']
            start_df_row['perquery_serps'] = curr_row['serps_num_session'] / curr_row['queries_num_session']
            #
            # start_df_row['perquery_queryterms'] = len(curr_row['queryterms_session']) / curr_row['queries_num_session']
            # start_df_row['perquery_queryterms_nonstop'] = len(curr_row['queryterms_nonstop_session']) / curr_row[
            #     'queries_num_session']
            # start_df_row['perquery_queryterms_unique'] = len(curr_row['queryterms_unique_session']) / curr_row[
            #     'queries_num_session']
            # start_df_row['perquery_queryterms_uniquenonstop'] = len(curr_row['queryterms_uniquenonstop_session']) / \
            #                                                     curr_row[
            #                                                         'queries_num_session']
            #
            # start_df_row['querydiversity_termpool'] = len(curr_row['queryterms_unique_session']) / len(
            #     curr_row['queryterms_session']) if len(curr_row['queryterms_session']) > 0 else 0
            # start_df_row['querydiversity_termpool_nonstop'] = len(curr_row['queryterms_uniquenonstop_session']) / len(
            #     curr_row['queryterms_nonstop_session']) if len(curr_row['queryterms_nonstop_session']) > 0 else 0
            #
            # clickindices = []
            # for q in curr_row['serp_clickindices_session']:
            #     for s in q:
            #         for click in s:
            #             clickindices += [click]
            # start_df_row['rank_maxclick_session'] = max(clickindices) if len(clickindices) > 0 else 0
            # start_df_row['rank_meanclick_session'] = np.mean(clickindices) if len(clickindices) > 0 else 0
            # start_df_row['max_viewed_rank_session'] = curr_row['max_viewed_rank_session']
            #
            # markov_action_types = ['start', 'query', 'serp', 'page']
            # markov_actions_total = 0
            # for a1 in markov_action_types:
            #     for a2 in markov_action_types:
            #         start_df_row['markov_actions_count_%s_%s' % (a1, a2)] = curr_row[
            #             'markov_actions_count_%s_%s' % (a1, a2)]
            #         markov_actions_total += curr_row['markov_actions_count_%s_%s' % (a1, a2)]
            # for a1 in markov_action_types:
            #     for a2 in markov_action_types:
            #         start_df_row['markov_actions_percent_%s_%s' % (a1, a2)] = curr_row['markov_actions_count_%s_%s' % (
            #         a1, a2)] / markov_actions_total if markov_actions_total > 0 else 0
            #
            # start_df_row['questionwords_num_session'] = curr_row['questionwords_num_session']
            # start_df_row['conjunctionwords_num_session'] = curr_row['conjunctionwords_num_session']
            # start_df_row['linkingverbs_num_session'] = curr_row['linkingverbs_num_session']
            # start_df_row['preposition_num_session'] = curr_row['preposition_num_session']
            #
            start_df_row['queryterms_num_session'] = len(curr_row['queryterms_session'])
            start_df_row['queryterms_numnonstop_session'] = len(curr_row['queryterms_nonstop_session'])
            # start_df_row['queryterms_numunique_session'] = len(curr_row['queryterms_unique_session'])
            # start_df_row['queryterms_numuniquenonstop_session'] = len(curr_row['queryterms_uniquenonstop_session'])
            #
            # reformulation_total = 0
            # for refType in reformulations.getReformulationTypes():
            #     start_df_row['reformulation_count_%s' % refType] = curr_row['reformulation_count_%s' % refType]
            #     reformulation_total += curr_row['reformulation_count_%s' % refType]
            # for refType in reformulations.getReformulationTypes():
            #     start_df_row['reformulation_percent_%s' % refType] = curr_row[
            #                                                              'reformulation_count_%s' % refType] / reformulation_total if reformulation_total > 0 else 0
            #
            start_df_row['start_action'] = True
            # start_df_row['local_timestamp'] = curr_row['local_timestamp']
            #
            # start_df_row['noclicks_queries_session'] = len(
            #     [serp_click_counts for serp_click_counts in curr_row['serp_clicks_session'] if
            #      np.sum(serp_click_counts) == 0])
            #
            start_df_row['elapsed_seconds'] = (curr_row['local_timestamp'] - curr_row['start_time'])
            # start_df_row['elapsed_minutes'] = (curr_row['local_timestamp'] - curr_row['start_time']) / 60.0
            start_df_row['query_lengths_mean'] = np.mean(curr_row['query_lengths_session'])

            # start_df_row['query_lengths_mean_nonstop'] = np.mean(curr_row['query_lengths_nonstop_session'])
            # start_df_row['query_lengths_min'] = np.min(curr_row['query_lengths_session'])
            # start_df_row['query_lengths_min_nonstop'] = np.min(curr_row['query_lengths_nonstop_session'])
            # start_df_row['query_lengths_max'] = np.max(curr_row['query_lengths_session'])
            # start_df_row['query_lengths_max_nonstop'] = np.max(curr_row['query_lengths_nonstop_session'])
            # start_df_row['query_lengths_var'] = np.var(curr_row['query_lengths_session'])
            # start_df_row['query_lengths_var_nonstop'] = np.var(curr_row['query_lengths_nonstop_session'])
            start_df_row['query_lengths_range'] = np.max(curr_row['query_lengths_session']) - np.min(
                curr_row['query_lengths_session'])
            #
            start_df_row['dwelltimes_total_content_session'] = np.sum(curr_row['dwelltimes_content_session']) if len(
                curr_row['dwelltimes_content_session']) >= 1 else 0
            start_df_row['dwelltimes_total_serp_session'] = np.sum(curr_row['dwelltimes_serp_session']) if len(
                curr_row['dwelltimes_serp_session']) >= 1 else 0
            # start_df_row['dwelltimes_percent_content_session'] = np.sum(curr_row['dwelltimes_content_session']) / (
            #     np.sum(curr_row['dwelltimes_content_session']) + np.sum(curr_row['dwelltimes_serp_session'])) if (len(
            #     curr_row['dwelltimes_content_session']) + len(curr_row['dwelltimes_serp_session'])) >= 1 else 0
            start_df_row['dwelltimes_percent_serp_session'] = np.sum(curr_row['dwelltimes_serp_session']) / (
                np.sum(curr_row['dwelltimes_content_session']) + np.sum(curr_row['dwelltimes_serp_session'])) if (len(
                curr_row['dwelltimes_content_session']) + len(curr_row['dwelltimes_serp_session'])) >= 1 else 0

            start_df_row['perquery_dwelltimes_total_serp_session'] = start_df_row['dwelltimes_total_serp_session'] / \
                                                                     start_df_row['queries_num_session'] if \
                start_df_row['queries_num_session'] > 0 else 0

            start_df_row['perquery_dwelltimes_total_content_session'] = start_df_row[
                                                                             'dwelltimes_total_content_session'] / \
                                                                        start_df_row['queries_num_session'] if \
                start_df_row['queries_num_session'] > 0 else 0

            start_df_row['dwelltimes_mean_content_session'] = np.mean(curr_row['dwelltimes_content_session']) if len(
                curr_row['dwelltimes_content_session']) >= 1 else 0
            # start_df_row['dwelltimes_mean_serp_session'] = np.mean(curr_row['dwelltimes_serp_session']) if len(
            #     curr_row['dwelltimes_serp_session']) >= 1 else 0

        if prev_row is not None:
            finish_df_row['session_num'] = prev_row['session_num']
            finish_df_row['action_type'] = prev_row['action_type']
            finish_df_row['start_time'] = prev_row['start_time']
            finish_df_row['interaction_num'] = prev_row['interaction_num']
            finish_df_row['total_actions_count'] = prev_row['total_actions_count']

            finish_df_row['pages_num_session'] = prev_row['pages_num_session']
            finish_df_row['pages_numunique_session'] = len(set(prev_row['urls_session']))
            finish_df_row['queries_num_session'] = prev_row['queries_num_session']
            finish_df_row['serps_num_session'] = prev_row['serps_num_session']
            # # finish_df_row['clicks_num_session'] = prev_row['clicks_num_session']
            #
            finish_df_row['perquery_pages'] = prev_row['pages_num_session'] / prev_row['queries_num_session']
            finish_df_row['perquery_serps'] = prev_row['serps_num_session'] / prev_row['queries_num_session']
            #
            # finish_df_row['perquery_queryterms'] = len(prev_row['queryterms_session']) / prev_row['queries_num_session']
            # finish_df_row['perquery_queryterms_nonstop'] = len(prev_row['queryterms_nonstop_session']) / prev_row[
            #     'queries_num_session']
            # finish_df_row['perquery_queryterms_unique'] = len(prev_row['queryterms_unique_session']) / prev_row[
            #     'queries_num_session']
            # finish_df_row['perquery_queryterms_uniquenonstop'] = len(prev_row['queryterms_uniquenonstop_session']) / \
            #                                                      prev_row[
            #                                                          'queries_num_session']
            #
            # finish_df_row['querydiversity_termpool'] = len(prev_row['queryterms_unique_session']) / len(
            #     prev_row['queryterms_session']) if len(prev_row['queryterms_session']) > 0 else 0
            # finish_df_row['querydiversity_termpool_nonstop'] = len(prev_row['queryterms_uniquenonstop_session']) / len(
            #     prev_row['queryterms_nonstop_session']) if len(prev_row['queryterms_nonstop_session']) > 0 else 0
            #
            # finish_df_row['questionwords_num_session'] = prev_row['questionwords_num_session']
            # finish_df_row['conjunctionwords_num_session'] = prev_row['conjunctionwords_num_session']
            # finish_df_row['linkingverbs_num_session'] = prev_row['linkingverbs_num_session']
            # finish_df_row['preposition_num_session'] = prev_row['preposition_num_session']
            #
            finish_df_row['queryterms_num_session'] = len(prev_row['queryterms_session'])
            finish_df_row['queryterms_numnonstop_session'] = len(prev_row['queryterms_nonstop_session'])
            # finish_df_row['queryterms_numunique_session'] = len(prev_row['queryterms_unique_session'])
            # finish_df_row['queryterms_numuniquenonstop_session'] = len(prev_row['queryterms_uniquenonstop_session'])
            #
            # reformulation_total = 0
            # for refType in reformulations.getReformulationTypes():
            #     finish_df_row['reformulation_count_%s' % refType] = prev_row['reformulation_count_%s' % refType]
            #     reformulation_total += prev_row['reformulation_count_%s' % refType]
            # for refType in reformulations.getReformulationTypes():
            #     finish_df_row['reformulation_percent_%s' % refType] = prev_row[
            #                                                               'reformulation_count_%s' % refType] / reformulation_total if reformulation_total > 0 else 0
            #
            # markov_action_types = ['start', 'query', 'serp', 'page']
            # markov_actions_total = 0
            # for a1 in markov_action_types:
            #     for a2 in markov_action_types:
            #         finish_df_row['markov_actions_count_%s_%s' % (a1, a2)] = prev_row[
            #             'markov_actions_count_%s_%s' % (a1, a2)]
            #         markov_actions_total += prev_row['markov_actions_count_%s_%s' % (a1, a2)]
            #
            # for a1 in markov_action_types:
            #     for a2 in markov_action_types:
            #         finish_df_row['markov_actions_percent_%s_%s' % (a1, a2)] = prev_row['markov_actions_count_%s_%s' % (
            #             a1, a2)] / markov_actions_total if markov_actions_total > 0 else 0
            #
            # # finish_df_row['serps_per_query'] = prev_row['serps_num_session']/prev_row['queries_num_session']
            finish_df_row['start_action'] = False
            # finish_df_row['local_timestamp'] = prev_row['local_timestamp']
            #
            finish_df_row['elapsed_seconds'] = (prev_row['local_timestamp'] - prev_row['start_time']) if prev_row is not None else 0
            # finish_df_row['elapsed_minutes'] = (prev_row['local_timestamp'] - prev_row['start_time']) / 60.0 if prev_row is not None else 0
            #
            finish_df_row['query_lengths_mean'] = np.mean(prev_row['query_lengths_session'])
            # finish_df_row['query_lengths_mean_nonstop'] = np.mean(prev_row['query_lengths_nonstop_session'])
            # finish_df_row['query_lengths_min'] = np.min(prev_row['query_lengths_session'])
            # finish_df_row['query_lengths_min_nonstop'] = np.min(prev_row['query_lengths_nonstop_session'])
            # finish_df_row['query_lengths_max'] = np.max(prev_row['query_lengths_session'])
            # finish_df_row['query_lengths_max_nonstop'] = np.max(prev_row['query_lengths_nonstop_session'])
            finish_df_row['query_lengths_range'] = np.max(prev_row['query_lengths_session']) - np.min(prev_row['query_lengths_session'])
            # finish_df_row['query_lengths_var'] = np.var(prev_row['query_lengths_session'])
            # finish_df_row['query_lengths_var_nonstop'] = np.var(prev_row['query_lengths_nonstop_session'])
            #
            # finish_df_row['noclicks_queries_session'] = len(
            #     [serp_click_counts for serp_click_counts in prev_row['serp_clicks_session'] if
            #      np.sum(serp_click_counts) == 0])
            #
            # clickindices = []
            # for q in prev_row['serp_clickindices_session']:
            #     for s in q:
            #         for click in s:
            #             clickindices += [click]
            # finish_df_row['rank_maxclick_session'] = max(clickindices) if len(clickindices) > 0 else 0
            # finish_df_row['rank_meanclick_session'] = np.mean(clickindices) if len(clickindices) > 0 else 0
            # finish_df_row['max_viewed_rank_session'] = prev_row['max_viewed_rank_session']
            #
            finish_df_row['dwelltimes_mean_content_session'] = np.mean(prev_row['dwelltimes_content_session']) if len(
                prev_row['dwelltimes_content_session']) >= 1 else 0

            finish_df_row['dwelltimes_total_content_session'] = np.sum(prev_row['dwelltimes_content_session']) if len(
                prev_row['dwelltimes_content_session']) >= 1 else 0
            finish_df_row['dwelltimes_total_serp_session'] = np.sum(prev_row['dwelltimes_serp_session']) if len(
                prev_row['dwelltimes_serp_session']) >= 1 else 0
            #
            # finish_df_row['dwelltimes_percent_content_session'] = np.sum(prev_row['dwelltimes_content_session']) / (
            #     np.sum(prev_row['dwelltimes_content_session']) + np.sum(prev_row['dwelltimes_serp_session'])) if (len(
            #     prev_row['dwelltimes_content_session']) + len(prev_row['dwelltimes_serp_session'])) >= 1 else 0
            finish_df_row['dwelltimes_percent_serp_session'] = np.sum(prev_row['dwelltimes_serp_session']) / (
                np.sum(prev_row['dwelltimes_content_session']) + np.sum(prev_row['dwelltimes_serp_session'])) if (len(
                prev_row['dwelltimes_content_session']) + len(prev_row['dwelltimes_serp_session'])) >= 1 else 0

            finish_df_row['perquery_dwelltimes_total_serp_session'] = finish_df_row['dwelltimes_total_serp_session'] / \
                                                                      finish_df_row['queries_num_session'] if \
                finish_df_row['queries_num_session'] > 0 else 0

            finish_df_row['perquery_dwelltimes_total_content_session'] = finish_df_row['dwelltimes_total_content_session'] / \
                                                                      finish_df_row['queries_num_session'] if \
                finish_df_row['queries_num_session'] > 0 else 0
            #
            # finish_df_row['dwelltimes_mean_serp_session'] = np.mean(prev_row['dwelltimes_serp_session']) if len(
            #     prev_row['dwelltimes_serp_session']) >= 1 else 0

        if prev_row is None:
            return [start_df_row]
        elif curr_row is None:
            return [finish_df_row]
        else:
            return [finish_df_row, start_df_row]

    def _get_dwelltime_info(self,curr_row, prev_row):
        dwell_time_type = 'content' if prev_row['action_type'] == 'page' else 'serp'
        duration = float(curr_row['local_timestamp']) - float(prev_row['local_timestamp'])
        # print("DURATION")
        # print({'type': dwell_time_type, 'duration': duration})
        return {'type': dwell_time_type, 'duration': duration}

    def process_log(self):
        # BEGIN POPULATING UNIQUENESS DICTS
        userID_none = -1
        for session in self.root:
            if session.tag == 'session':
                userID = None
                questionID = None
                if session.attrib['userid'] != 'NA':
                    userID = int(session.attrib['userid'])
                else:
                    userID = userID_none
                    userID_none -= 1

                for topic in session.iter('topic'):
                    if topic.tag == 'topic':
                        questionID = int(topic.attrib['num'])

                for interaction in list(session.iter('interaction')) + list(session.iter('currentquery')):
                    query = interaction.find('query').text.lower()
                    query_terms = query.split()

                    urls = []
                    if interaction.find('clicked') is not None:
                        for clicked in interaction.findall('clicked'):
                            for click in clicked.findall('click'):
                                doc = click.find('docno').text
                                urls += [doc]

                    # if self.topicqueries_to_users.get((questionID, query), None) is None:
                    #     self.topicqueries_to_users[(questionID, query)] = set([userID])
                    # else:
                    #     self.topicqueries_to_users[(questionID, query)] |= set([userID])
                    #
                    # for term in query_terms:
                    #     if self.topicqueryterms_to_users.get((questionID, term), None) is None:
                    #         self.topicqueryterms_to_users[(questionID, term)] = set([userID])
                    #     else:
                    #         self.topicqueryterms_to_users[(questionID, term)] |= set([userID])
                    #
                    # for u in urls:
                    #     if self.topicurls_to_users.get((questionID, u), None) is None:
                    #         self.topicurls_to_users[(questionID, u)] = set([userID])
                    #     else:
                    #         self.topicurls_to_users[(questionID, u)] |= set([userID])

        # END POPULATING UNIQUENESS DICTS
        VALID_TOPICS = [i for i in range(1, 61)]
        VALID_USERTOPICS = dict()

        # Users and completed users
        df = []
        facets_eda_df = []
        overview_df = []
        query_length_means = []
        query_length_variances = []
        userID_none = -1
        for session in self.root:
            prev_row = None
            row = {}
            if session.tag == 'session':
                userID = None
                questionID = None
                trec_interactions_count = 0
                total_actions_count = 0
                if session.attrib['userid'] != 'NA':
                    userID = int(session.attrib['userid'])
                else:
                    userID = userID_none
                    userID_none -= 1

                for topic in session.iter('topic'):
                    if topic.tag == 'topic':
                        questionID = int(topic.attrib['num'])

                # Metadata
                row['userID'] = userID
                row['start_time'] = float(session.attrib['starttime'])
                row['questionID'] = questionID
                row['session_num'] = int(session.attrib['num'])
                print('userID', userID, 'session_num', row['session_num'])
                row['facet_trec2014tasktype'] = self.topic_to_trectasktype[questionID]
                row['facet_goal'] = self.topic_to_goal[questionID]
                row['facet_product'] = self.topic_to_product[questionID]
                # row['facet_complexity'] = topic_to_complexity[questionID]
                # row['facet_complexity_merged'] = topic_to_complexity_merged[questionID]
                # row['facet_complexity_create'] = topic_to_complexity_create[questionID]


                # Session data
                row['pages_num_session'] = 0
                row['queries_num_session'] = 0
                row['queries_session'] = []
                row['queryterms_unique_session'] = set()
                row['queryterms_uniquenonstop_session'] = set()
                row['queryterms_session'] = []
                row['queryterms_nonstop_session'] = []
                row['urls_session'] = []
                row['serps_num_session'] = 0
                row['clicks_num_session'] = 0

                row['query_lengths_session'] = []
                row['query_lengths_nonstop_session'] = []
                row['queryterms_unique_session'] = set()
                row['query_chars_session'] = 0
                row['query_chars_nonstop_session'] = 0

                row['questionwords_num_session'] = 0
                row['conjunctionwords_num_session'] = 0
                row['linkingverbs_num_session'] = 0
                row['preposition_num_session'] = 0
                row['max_viewed_rank_session'] = 0

                # row['queries_session'] = []

                row['dwelltimes_content_session'] = []
                row['dwelltimes_serp_session'] = []
                row['serp_clicks_session'] = []
                row['serp_clickindices_session'] = []
                row['query_relevance_judgments_session'] = []
                row['serp_resultsnippets_session'] = []
                row['serp_resulturls_session'] = []

                for refType in reformulations.getReformulationTypes():
                    row['reformulation_count_%s' % refType] = 0

                # For each interaction:
                # Query: get the query, get the timestamp.  Action is query.  Increment number of queries.  Get query statistics.  Submit
                # Click: get the page source,

                total_interactions = len(list(session.iter('interaction')) + list(session.iter('currentquery')))
                serpnum = 1

                action_types = ['start', 'query', 'serp', 'page']
                for a1 in action_types:
                    for a2 in action_types:
                        row['markov_actions_count_%s_%s' % (a1, a2)] = 0

                for interaction in list(session.iter('interaction')) + list(session.iter('currentquery')):
                    trec_interactions_count += 1
                    total_actions_count += 1
                    row['action_type'] = 'query'
                    row['data'] = interaction.find('query').text
                    row['query'] = ''.join(interaction.find('query').text.split())
                    row['local_timestamp'] = float(interaction.attrib['starttime'])

                    if trec_interactions_count < total_interactions:
                        row['interaction_num'] = int(interaction.attrib['num'])
                    elif interaction.tag == 'interaction':
                        row['interaction_num'] = int(interaction.attrib['num'])
                    else:
                        row['interaction_num'] = prev_row['interaction_num'] + 1

                    row['serps_num_session'] += 1
                    if interaction.attrib.get('type', None) is None or interaction.attrib['type'] == 'reformulate':

                        if interaction.find('query').text.lower() in row['queries_session']:
                            row['action_type'] = 'serp'
                        row['queries_session'] += [interaction.find('query').text.lower()]

                        row['questionwords_num_session'] += len(
                            [w for w in interaction.find('query').text.lower().split() if w in self.questionwords])
                        row['conjunctionwords_num_session'] += len(
                            [w for w in interaction.find('query').text.lower().split() if w in self.conjunctionwords])
                        row['linkingverbs_num_session'] += len(
                            [w for w in interaction.find('query').text.lower().split() if w in self.linkingverbs])
                        row['preposition_num_session'] += len(
                            [w for w in interaction.find('query').text.lower().split() if w in self.prepositions])

                        if len([w for w in interaction.find('query').text.lower().split() if
                                w in self.questionwords]) > 0:
                            print(
                                [w for w in interaction.find('query').text.lower().split() if w in self.questionwords])

                        row['queryterms_session'] += interaction.find('query').text.lower().split()
                        row['queryterms_nonstop_session'] += [term for term in
                                                              interaction.find('query').text.lower().split() if
                                                              term not in self.stopwords]

                        row['queryterms_unique_session'] |= set(interaction.find('query').text.lower().split())
                        row['queryterms_uniquenonstop_session'] |= set(
                            [term for term in interaction.find('query').text.lower().split() if
                             term not in self.stopwords])

                        row['queries_num_session'] = len(row['queries_session'])
                        row['serp_clicks_session'] += [[0]]  # 2d array.  [qnum][serpnum]
                        row['serp_clickindices_session'] += [[[]]]  # 2d array.  [qnum][serpnum]->clickarray
                        row['serp_resultsnippets_session'] += [[]]
                        row['serp_resulturls_session'] += [[]]
                        serpnum = 1
                        if prev_row is None:
                            row['reformulation'] = reformulations.classifyReformulation(None, row['query'])
                        else:
                            row['reformulation'] = reformulations.classifyReformulation(prev_row['query'], row['query'])

                        for refType in reformulations.getReformulationTypes():
                            row['reformulation_count_%s' % refType] += int(row['reformulation'] == refType)

                        row['query_chars_session'] += len(interaction.find('query').text)
                        row['query_chars_nonstop_session'] = len(
                            ' '.join([term for term in interaction.find('query').text if term not in self.stopwords]))
                    else:
                        row['action_type'] = 'serp'
                        row['serp_clicks_session'][-1] += [0]
                        row['serp_clickindices_session'][-1] += [[]]  # 2d array.  [qnum][serpnum]->clickarray
                        row['serp_resultsnippets_session'] += []
                        row['serp_resulturls_session'] += []
                        serpnum += 1
                        if interaction.attrib['type'] != 'page':
                            print("TYPE", interaction.attrib['type'])
                            exit()

                    row['total_actions_count'] = total_actions_count
                    if prev_row is None:
                        row['markov_actions_count_%s_%s' % ('start', row['action_type'])] += 1
                    else:
                        row['markov_actions_count_%s_%s' % (prev_row['action_type'], row['action_type'])] += 1

                    # row['dwelltimes_content_segment'] = []
                    # row['dwelltimes_serp_segment'] = []
                    # row['pages_num_segment'] = 0
                    # # if interaction.tag=='currentquery' or (interaction.tag=='interaction' and interaction.attrib['starttime']):


                    # row['session_prec10'] = []



                    if prev_row is not None:
                        dwelltime_info = self._get_dwelltime_info(row, prev_row)
                        row['dwelltimes_%s_session' % dwelltime_info['type']] += [dwelltime_info['duration']]
                        # row['dwelltimes_%s_segment' % dwelltime_info['type']] += [dwelltime_info['duration']]
                    #
                    #
                    #
                    row['query_length'] = len(interaction.find('query').text.split())
                    row['query_length_nonstop'] = len(
                        [term for term in interaction.find('query').text.split() if term not in self.stopwords])

                    # If a new query
                    if interaction.attrib.get('type', None) is None or interaction.attrib['type'] == 'reformulate':
                        row['query_lengths_session'] += [row['query_length']]
                        row['query_lengths_nonstop_session'] += [row['query_length_nonstop']]
                        row['queryterms_unique_session'] |= set(interaction.find('query').text.split())
                        row['queryterms_uniquenonstop_session'] |= set(
                            [term for term in interaction.find('query').text.split() if term not in self.stopwords])

                    if trec_interactions_count == total_interactions:
                        query_length_means += [np.mean(row['query_lengths_session'])]
                        query_length_variances += [np.var(row['query_lengths_session'])]
                    #
                    # row['queries_session'] += [interaction.find('query').text]


                    result_urls = []
                    result_ranks = []
                    result_snippets = []
                    result_docids = []
                    result_relevances = []
                    for resultslist in interaction.findall('results'):
                        for result in resultslist.findall('result'):
                            result_urls += [result.find('url').text]
                            result_ranks += [int(result.attrib['rank'])]
                            result_snippets += [
                                result.find('snippet').text if result.find('snippet').text is not None else '']
                            result_docids += [result.find('clueweb12id').text]

                    row['max_viewed_rank_session'] = max(row['max_viewed_rank_session'],
                                                         max(result_ranks) if len(result_ranks) > 0 else 0)

                    row['serp_resultsnippets_session'][-1] += result_snippets
                    row['serp_resulturls_session'][-1] += result_urls

                    # print([len(r) for r in  row['serp_resulturls_session']])

                    for docid in result_docids:
                        if self.judgmentsbytopic.get(questionID, None) is None or self.judgmentsbytopic[questionID].get(docid,
                                                                                                              None) is None:
                            result_relevances += [0]
                        else:
                            result_relevances += [max([self.judgmentsbytopic[questionID][docid], 0])]

                    # if interaction.attrib.get('type', None) is None or interaction.attrib['type'] == 'reformulate':
                    #     print(row['query_relevance_judgments_session'])
                    #     row['query_relevance_judgments_session'] += [result_relevances]
                    # else:
                    #     print(row['query_relevance_judgments_session'])
                    #     row['query_relevance_judgments_session'][-1] += result_relevances
                    # print(row['query_relevance_judgments_session'])




                    # row['query_prec10'] = float(sum(result_relevances)) / len(result_relevances) if len(
                    #     result_relevances) > 0 else 0
                    # row['session_prec10'] += [row['query_prec10']]


                    df += self._make_df_rows(row, prev_row)
                    prev_row = copy.deepcopy(row)

                    if interaction.find('clicked') is not None:
                        for clicked in interaction.findall('clicked'):
                            for click in clicked.findall('click'):
                                row['urls_session'] += [click.find('docno').text]
                                total_actions_count += 1
                                row['action_type'] = 'page'
                                # row['data'] = result_urls[int(click.attrib['num']) - 1]
                                row['local_timestamp'] = float(click.attrib['starttime'])
                                row['total_actions_count'] = total_actions_count
                                if prev_row is None:
                                    row['markov_actions_count_%s_%s' % ('start', row['action_type'])] += 1
                                else:
                                    row['markov_actions_count_%s_%s' % (
                                        prev_row['action_type'], row['action_type'])] += 1

                                # row['pages_num_segment'] += 1
                                row['pages_num_session'] += 1
                                row['clicks_num_session'] += 1

                                row['serp_clicks_session'][-1][-1] += 1

                                docid = click.find('docno').text
                                if docid in result_docids:
                                    clickrank = result_ranks[result_docids.index(docid)]
                                    # print("CLICKINDEX",clickrank)
                                    row['serp_clickindices_session'][-1][-1] += [int(clickrank)]
                                    # print(row['serp_clickindices_session'])

                                # row['queries_clicked_ranks'] += [int(click.attrib['num'])]
                                # row['queries_max_clicked_ranks'][-1] = max(
                                #     [row['queries_max_clicked_ranks'][-1], int(click.attrib['num'])])
                                #
                                if prev_row is not None:
                                    dwelltime_info = self._get_dwelltime_info(row, prev_row)
                                    row['dwelltimes_%s_session' % dwelltime_info['type']] += [
                                        dwelltime_info['duration']]
                                    # row['dwelltimes_%s_segment' % dwelltime_info['type']] += [dwelltime_info['duration']]

                                df += self._make_df_rows(row, prev_row)
                                prev_row = copy.deepcopy(row)

                overview_df += [{'user_num': userID, 'session_num': row['session_num'], 'topic_num': questionID,
                                 'COUNT(trec_interactions)': trec_interactions_count,
                                 'COUNT(total_actions)': total_actions_count,
                                 'queries_num_session': row['queries_num_session']}]

        pd.DataFrame(df).to_csv(self.output_csv)
        # pd.DataFrame(overview_df).to_csv('../../data/interim/trec2014_sessions_overview.csv')
        print("MEAN OF MEAN", np.mean(query_length_means))
        print("VAR OF MEAN", np.var(query_length_means))
        print("MEAN OF VAR", np.mean(query_length_variances))
        print("VAR OF VAR", np.var(query_length_variances))







if __name__ == '__main__':
    logger = NSFLogParser(output_csv='/Users/Matt/Desktop/sigir_features_trec.csv')
    logger.process_log()
    exit()

    # print("BEGIN SESSIONS PER USER")
    # sessions_per_user_df = []
    # for (n, group) in pd.DataFrame(overview_df).groupby('user_num'):
    #     if n < 1:
    #         continue
    #     sessions_per_user_df += [{'user_num': n, 'num_sessions': len(group.index)}]
    #     pd.DataFrame(sessions_per_user_df).to_csv('../../data/interim/trec2014_sessions_per_user_nonnegativeuserid.csv')
    #
    # print("BEGIN USERS PER TOPIC")
    # users_per_topic_df = []
    # for (n, group) in pd.DataFrame(overview_df).groupby('topic_num'):
    #     users_per_topic_df += [
    #         {'topic_num': n, 'num_sessions': len(set(group[group['user_num'] >= 1]['user_num'].tolist()))}]
    #     pd.DataFrame(users_per_topic_df).to_csv('../../data/interim/trec2014_users_per_topic_nonnegativeuserid.csv')

        #
        # facet_eda_df = pd.DataFrame(df)
        # facet_eda_df = facet_eda_df[['userID','questionID','action_type','data','interaction_num','facet_goal','facet_product']]
        #
        # facet_eda_df = facet_eda_df[facet_eda_df['userID']>=1]
        #
        # print(facet_eda_df)
        #
        # for (n,group) in facet_eda_df.groupby(['facet_goal']):
        #     d = []
        #     for (n2,user_group) in group.groupby(['userID','questionID']):
        #         d += [user_group]
        #         d += [pd.DataFrame([{
        #         'userID':'--',
        #         'questionID':'--',
        #         'action_type':'--',
        #         'data':'--',
        #         'interaction_num':'--',
        #         'facet_goal':'--',
        #         'facet_product':'--'
        #         }])]
        #         d += [pd.DataFrame([{
        #         'userID':'--',
        #         'questionID':'--',
        #         'action_type':'--',
        #         'data':'--',
        #         'interaction_num':'--',
        #         'facet_goal':'--',
        #         'facet_product':'--'
        #         }])]
        #
        #     pd.concat(d).to_csv('/Users/Matt/Desktop/FACET EDA/Goal (Search) - %s/trec2014-%s.csv'%(n,n))
        #
        #
        # for (n,group) in facet_eda_df.groupby(['facet_product']):
        #     d = []
        #     for (n2,user_group) in group.groupby(['userID','questionID']):
        #         d += [user_group]
        #         d += [pd.DataFrame([{
        #         'userID':'--',
        #         'questionID':'--',
        #         'action_type':'--',
        #         'data':'--',
        #         'interaction_num':'--',
        #         'facet_goal':'--',
        #         'facet_product':'--'
        #         }])]
        #         d += [pd.DataFrame([{
        #         'userID':'--',
        #         'questionID':'--',
        #         'action_type':'--',
        #         'data':'--',
        #         'interaction_num':'--',
        #         'facet_goal':'--',
        #         'facet_product':'--'
        #         }])]
        #
        #     pd.concat(d).to_csv('/Users/Matt/Desktop/FACET EDA/Product (Search) - %s/trec2014-%s.csv'%(n,n))
        #
        # d = []
        # d1 = []
        # for (n2,user_group) in facet_eda_df.groupby(['userID','questionID']):
        #     (userID,questionID) = n2
        #     d += [user_group]
        #     d1 += [{'userID':userID,'questionID':questionID,'pages':'','queries':'','facet_product':user_group['facet_product'].tolist()[0],'facet_goal':user_group['facet_goal'].tolist()[0],}]
        #     d += [pd.DataFrame([{
        #     'userID':'--',
        #     'questionID':'--',
        #     'current_data':'--',
        #     'pq_current_action_type':'--',
        #     'action_type':'--',
        #     'data':'--',
        #     'current_tab':'--',
        #
        #     'pq_current_action_count':'--',
        #     'facet_goal':'--',
        #     'facet_product':'--'
        #     }])]
        #     d += [pd.DataFrame([{
        #     'userID':'--',
        #     'questionID':'--',
        #     'current_data':'--',
        #     'pq_current_action_type':'--',
        #     'action_type':'--',
        #     'data':'--',
        #     'current_tab':'--',
        #
        #     'pq_current_action_count':'--',
        #     'facet_goal':'--',
        #     'facet_product':'--'
        #     }])]
        #
        # pd.concat(d).to_csv('/Users/Matt/Desktop/FACET EDA/All Data/trec2014-All.csv')
        # pd.DataFrame(d1).to_csv('/Users/Matt/Desktop/FACET EDA/All Data/trec2014-Comments.csv')



