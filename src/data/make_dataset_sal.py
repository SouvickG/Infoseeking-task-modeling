from __future__ import print_function

import copy
import sys
from urlparse import urlparse

import mysql.connector
import numpy as np
import pandas as pd

sys.path.insert(1, '../utils')
from nltk.corpus import stopwords
from stop_words import get_stop_words


def action_states_for_user_task_decorator(func):
    argnames = func.func_code.co_varnames[:func.func_code.co_argcount]
    fname = func.func_name

    def wrapper(*args, **kwargs):
        print(fname, "(", ', '.join(
            '%s=%r' % entry
            for entry in zip(argnames, args[:len(argnames)]) + [("args", list(args[len(argnames):]))] + [
                ("kwargs", kwargs)]) + ")")
        returnval = func(*args, **kwargs)
        print("action_states_for_user_task_decorator - OUTPUT LENGTH: %d" % len(returnval))
        return returnval

    return wrapper


class SALLogParser:
    #TODO: Account for Etherpad
    #TODO: Account for Coagmento
    #Object information
    valid = False
    connection = None
    dbname = 'spring2017_learning'
    dbhost = 'localhost'
    dbuser = 'root'
    dbpwd = 'root'
    substring_coagmento = 'coagmento.org'
    substring_etherpad = 'coagmentopad.rutgers.edu'
    substrings_badurls = ['facebook.com', 'buzzfeed.com', 'rutgers.edu','mail.google.com','tumblr.com','youtube.com',
                          'acnestudios','pof.com','twitch.tv','biology','engineering','calendar.google.com','craigslist.com',
                          'glassdoor.com','humanbenchmark.com','imgur.com','reddit.com/r/nosleep','reddit.com/r/photoshopbattles',
                          'iflmylife.com','twitter.com','thesaurus.com','ebay.com','livepolish.com','amazon.com','instagram.com',
                          'mensa.org','cambridgebrainsciences.com','collegeboard.org','iqcomparisonite.com','putlockeris.org',
                          'radaris.com','rutgersonline.net','rutgers.edu','ecollege.com','watchcommunity.tv','watchseries.cx',
                          'weibo.com','wowmovie.me','acnestudioes.com','chanel.com','cookinglight.com','coastalliving.com',
                          'deviantart.com','foodandwine.com','geekrank.com','iflmylife.com','iqcomparisonite.com',
                          'musiciansfriend.com','preziouz.com','accounts.google.com','calendar.google.com','craigslist.org',
                          'docs.google.com','datemyschool.com','drive.google.com','hackensackuhn','minecraft.net',
                          'myaccount.google.com','myaccount.allstate.com','spotify.com','putlockeris','us.battle.net',
                          'dominos.com','epicgames.com','ezpassnj.com','linkedin.com','netflix.com',
                          'patreon.com','reddit.com/r/ComedyCemetery','reddit.com/r/FireEmblemHeroes','reddit.com/r/hearthstone',
                          'reddit.com/r/MakeUpAddiction','reddit.com/r/KirbyPasta','reddit.com/r/DocChamp','trend-chaser.com',
                          'taobao.com','storets.com','pixiemarket.com','codeacademy.com',
                          'adamsconcierge.com','boards.4chan.org','bzfd.it','vonvon.me','everynoise.com','holyspiritweb.org',
                          'diply.com','albinoblacksheel','cbs.com','iqcomparisonsite','siteadvisor.com','pearsoncmg.com',
                          'pearson.com','codeacademy.com','coursera.org','reddit.com/r/MakeupAddiction','reddit.com/r/Yogscast',
                          'reddit.com/r/explainlikeimfive','usertesting.com','codecademy.com','reddit.com/r/news','reddit.com/r/funny']

    #User/Session information
    completed_userIDs = [
        130,
        162,
        164,
        108,
        146,
        124,
        147,
        148,
        100,
        135,
        169,
        150,
        168,
        137,
        170,
        82,
        115,
        136,
        139,
        157,
        174,
        153,
        175,
        62,
        173,
        123,
        141,
        158,
        121,
        54,
        160
    ]
    completed_userID_to_participantID = {
        54:    'S005',
        82:    'S022',
        62:    'S008',
        100:    'S016',
        108:    'S006',
        115:    'S011',
        121:    'S024',
        123:    'S021',
        124:    'S029',
        130:    'S026',
        135:    'S031',
        136:    'S034',
        137:    'S033',
        139:    'S036',
        141:    'S039',
        146:    'S044',
        147:    'S045',
        148:    'S046',
        150:    'S048',
        153:    'S053',
        157:    'S052',
        158:    'S055',
        160:    'S057',
        162:    'S058',
        164:    'S059',
        168:    'S064',
        169:    'S063',
        170:    'S065',
        173:    'S070',
        174:    'S068',
        175:    'S069'
    }
    task_numbers = [1,2,3,4]
    tasknum_to_taskstageid = {
        1: 15,
        2: 25,
        3: 35,
        4: 45,
    }
    tasknum_to_aftertaskstageid = {
        1: 16,
        2: 26,
        3: 36,
        4: 46,
    }
    skip_sessions = [(135,1),(135, 2),(169,1),(169,2),(169,3),(169,4),(170,3),(175,3),(175,4),(62,2),(173,3)]  # (userID,taskNum of tasks to skip)

    #Action information
    actions_click = ['left-click', 'left-dblclick', 'middle-click', 'right-click', 'right-dblclick']
    actions_query = ['query and 1', 'query and 2', 'query and 3', 'query and 4', 'query']
    actions_query_transform = ['query and 1', 'query and 2', 'query and 3', 'query and 4']
    actions_bookmark_save = ['Save Bookmark']
    actions_bookmark_unsave = ['delete_bookmarks']
    actions_page = ['page']
    actions_tab = ['tabAdded', 'tabClosed', 'tabSelected']
    valid_actions = actions_click + actions_query + actions_bookmark_save + actions_bookmark_unsave + actions_page + actions_tab
    action_counts = dict()
    missing_action_counts = dict()
    action_map = None
    same_action_elapsed_time = 50  # milliseconds
    long_page_read_seconds = 17.45  # if greater than this, long page read

    #Page/Query Information
    viewmystuff_url = u'http://coagmento.org/fall2015intent/services/viewMyStuff.php?=true'
    pageid_to_url = None
    queryid_to_url = None
    page_data = None
    query_data = None
    invalid_queryIDs = None #TODO
    invalid_pageIDs = None # TODO: Don't worry about this for now. Next time
    invalid_queryIDs_fileloc = '/Users/Matt/Documents/Programming/GitHub/Repositories/task-modeling/data/interim/sal_bad_queries.csv'
    invalid_pageIDs_fileloc = None  # TODO: Don't worry about this for now. Next time
    long_read_errortime_seconds = 300.0
    long_queryread_errortime_seconds = 120.0

    auxiliary_datasets = None



    session_to_presearch = None
    questionnaire_columns_presearch = ['topic_knowledge','knowledge_help','topic_interest','seek_difficulty','understand_difficulty']
    session_to_postsearch = None
    questionnaire_columns_postsearch = ['taskcompletion_time','post_knowledge','help_priorknowledge','task_difficulty','infoseek_difficulty','understand_difficulty','interest_increase','knowledge_increase']
    session_to_demographic = None
    questionnaire_columns_demographic = ['searchexperience_years','search_perday','search_skills','english_readproficiency','english_writeproficiency']

    stopwords = set(stopwords.words('english') + get_stop_words('en'))

    stage_to_sessionnum = {
        1:1,
        5:1,
        7:1,
        10:1,
        15:1,
        16:1,
        17:1,
        20:2,
        25:2,
        26:2,
        27:2,
        30:3,
        35:3,
        36:3,
        37:3,
        40:4,
        45:4,
        46:4,
        47:4,
        50:4,
        295:4,
        300:4,
    }


    # Done
    def _sql_query(self, query):
        return pd.read_sql(query, con=self.connection)

    # Done
    def _assert_valid_instance(self):
        if not self.valid:
            print("Not a valid object.  Please create another instance of %s"%(self.__class__.__name__))
        assert self.valid

    # Done
    def _extract_questionnaire_data(self,table_name,column_names):
        questionnaire_dict = dict()
        for (n, row) in self._sql_query("SELECT * FROM %s"%(table_name,)).iterrows():
            questionnaire_dict[(row['userID'], self.stage_to_sessionnum[row['stageID']])] = dict()
            for colname in column_names:
                questionnaire_dict[(row['userID'], self.stage_to_sessionnum[row['stageID']])][colname] = row[colname]
        return questionnaire_dict

    # Done
    def _extract_idtourl_data(self,table_name):
        assert table_name in ['pages','queries']
        id_column = 'pageID' if table_name =='pages' else 'queryID'
        return dict(
            [
                (row[id_column], row['url'])
                for (n, row) in self._sql_query(
                "SELECT * from %s WHERE userID IN %s"%(table_name,str(tuple(self.completed_userIDs)))
            ).iterrows()
            ]
        )

    # Done
    # Extract queryID's of queries issued by user (versus duplicates)
    def _extract_issued_queryIDs(self, userID, stageID):
        urls = []
        queryIDs = []
        queries_df = self._sql_query(
            "SELECT url,queryID FROM queries WHERE userID=%d AND stageID=%d AND `localTimestamp`!=0 ORDER BY `localTimestamp`,queryID" % (
                userID, stageID))
        for (n, row) in queries_df.iterrows():
            url = row['url']
            queryID = row['queryID']
            if not url in urls:
                urls += [url]
                queryIDs += [queryID]
        return queryIDs


    #Done
    def _extract_pq_data(self,table_name):
        assert table_name in ['pages', 'queries']
        id_column = 'pageID' if table_name == 'pages' else 'queryID'
        pq_df = self._sql_query("SELECT * from %s WHERE userID IN %s"%(table_name,str(tuple(self.completed_userIDs))))

        pq_dict = dict()

        for (n,row) in pq_df.iterrows():
            pq_id = row[id_column]
            pq_dict[pq_id] = dict()
            pq_dict[pq_id]['type'] = 'page' if table_name=='pages' else 'query'
            pq_dict[pq_id]['url'] = row['url']
            pq_dict[pq_id]['title'] = row['title']
            pq_dict[pq_id]['source'] = row['source']
            if table_name=='queries':
                pq_dict[pq_id]['query'] = row['query']

        return pq_dict

    def _extract_invalid_queryIDs(self):
        invalid_queries_df = pd.read_csv(self.invalid_queryIDs_fileloc)
        return (invalid_queries_df['queryID'].tolist(), invalid_queries_df['url'].tolist())

    # Done
    def __init__(self, output_csv='./output.csv'):
        self.output_csv = output_csv
        self.auxiliary_datasets = None
        # TODO: Action Map
        self.valid = True
        self.action_map = {
            #     # pages+queries
            ('page',): self._pagequery_behavior,  # As you'd expect, probably, unless in close vicinity of other actions
            ('page', 'query'): self._pagequery_behavior,
            # # As you'd expect, probably, unless in close vicinity of other actions
            #     (u'page', u'page', u'query'): self._pagequery_behavior,
            #     # What's going on here? user80: weird time delay bug.  User 48 opened a new tab but an old tab was still being logged (thankfully previously existed in log).  user 82 had multiple windows.  Temp solution: 1) condense to as many pages as possible.  If still multiple pages: if an existing page, probably in a different window so ignore.  For the other, change current tab to new one.  Can't assume which one comes first.  Just assume the page
            #     (u'page', u'page', u'query', u'query'): self._pagequery_behavior,
            #     # What's going on here? user4: multiple windows, one is new. user3 all one window, but new page and some previously exist.  So clean those that aready exist.  If new, assume it's a new page.  Or rather, do the same behavior as with any other pages.  Another hidden tab.  Presumably no way to tell which one is first.  When in doubt, just assume current tab and assume the duplicate information is somewhere else.
            (u'page', u'page'): self._pagequery_behavior,  # TODO: both the same page? Verify that's the case
            #     # What's going on here? user82 has multiple windows open, one is in the background.  user55, also background tabs. user35 also has background tabs
            #     # viewMyStuff
            #     (u'page', u'page', u'page'): self._pagequery_behavior,  # What's going on here?  Probably same as above
            #     (u'page', u'page', u'page', u'page', u'query', u'query'): self._pagequery_behavior,
            #     # What's going on here? Probably same as above
            #     (u'page', u'page', u'page', u'query'): self._pagequery_behavior,
            # # What's going on here? Probably same as above
        }

        self.connection = mysql.connector.connect(host=self.dbhost, user=self.dbuser, password=self.dbpwd, database=self.dbname)

        self.session_to_presearch = self._extract_questionnaire_data('questionnaire_presearch',self.questionnaire_columns_presearch)
        self.session_to_postsearch = self._extract_questionnaire_data('questionnaire_postsearch',
                                                                     self.questionnaire_columns_postsearch)
        self.session_to_demographic = self._extract_questionnaire_data('questionnaire_demographic',
                                                                     self.questionnaire_columns_demographic)

        self.pageid_to_url = self._extract_idtourl_data('pages')
        self.queryid_to_url = self._extract_idtourl_data('queries')
        self.page_data = self._extract_pq_data('pages')
        self.query_data = self._extract_pq_data('queries')
        self.invalid_queryIDs,self.invalid_queryurls = self._extract_invalid_queryIDs()
        self.substrings_badurls += self.invalid_queryurls




    # Done
    def _get_session_metadata(self, userID, taskNum):
        # Note: All users have same ordering of tasks 1-2-3-4-5-6; no Latin Square
        participantID = self.completed_userID_to_participantID[userID]
        questionIDs = self._sql_query(
            "SELECT * FROM pages WHERE userID=%d AND questionID >=1 GROUP BY questionID ORDER BY `stageID` ASC" % userID)[
            'questionID'].tolist()
        print(questionIDs)
        # assert len(questionIDs) == 6
        questionID = questionIDs[taskNum - 1]
        stageID_task = self.tasknum_to_taskstageid[taskNum]
        stageID_posttask = self.tasknum_to_aftertaskstageid[taskNum]

        question_data = self._sql_query("SELECT * FROM questions_study WHERE questionID=%d" % questionID)
        task_product = question_data['product'].tolist()[0]
        task_goal = question_data['goal'].tolist()[0]
        print(userID,taskNum,stageID_task)
        taskstage_data = self._sql_query(
            "SELECT * FROM session_progress WHERE userID=%d AND stageID=%d" % (userID, stageID_task))
        posttaskstage_data = self._sql_query(
            "SELECT * FROM session_progress WHERE userID=%d AND stageID=%d" % (userID, stageID_posttask))
        question_starttime = taskstage_data['timestamp'].tolist()[0] * 1000
        question_endtime = posttaskstage_data['timestamp'].tolist()[0] * 1000
        task_topic = taskNum
        num_queries_total = len(self._extract_issued_queryIDs(userID,stageID_task))

        task_type = None
        if (task_product == 'Factual' and task_goal == 'Specific'):
            task_type = 'Known-Item'
        elif (task_product == 'Factual' and task_goal == 'Amorphous'):
            task_type = 'Known-Subject'
        elif (task_product == 'Intellectual' and task_goal == 'Specific'):
            task_type = 'Interpretive'
        else:
            task_type = 'Exploratory'

        to_return = {
            'startTimestamp': question_starttime,
            'endTimestamp': question_endtime,
            'userID': userID,
            'participantID': participantID,
            'questionID': questionID,
            'task_topic': task_topic,
            'facet_product': task_product,
            'task_type': task_type,
            'facet_goal': task_goal,
            'num_queries_total': num_queries_total,
        }
        to_return.update(self.session_to_presearch[(userID, taskNum)])
        to_return.update(self.session_to_postsearch[(userID, taskNum)])
        to_return.update(self.session_to_demographic[(userID, 1)])
        return to_return

    # Done
    # Input: Data frame from actions
    # Output: new frame of page/query actions, indicating actionID (from actions table), it of item, url, action type
    def _get_urls_ids_from_actions_frame(self, action_data):
        new_frame = []
        for (n, row) in action_data.iterrows():
            item_id = row['value']
            action = row['action']
            url = None
            if action == 'page':
                url = self.pageid_to_url[item_id]
            elif action == 'query':
                url = self.queryid_to_url[item_id]
            new_frame += [
                {'actionID': row['actionID'], 'pagequeryID': item_id, 'url': url, 'action': action}]
        return pd.DataFrame(new_frame)

    # Done
    def _get_urls_from_frame(self, data):
        ids = data['value'].tolist()
        actions = data['action'].tolist()
        urls = []
        for (i, act) in zip(ids, actions):
            if act == 'page':
                urls += [self.pageid_to_url[i]]
            elif act == 'query':
                urls += [self.queryid_to_url[i]]
        return urls


    # Done
    # Input: a page/query data frame
    # Output: Cleaned version of the frame
    # Cleaning:
    # 1) Removed duplicate URLs (in the case of page-query action, removes the page)
    # 2) Asserts
    def _clean_duplicates_from_pqframe(self, data):
        data_with_urls = self._get_urls_ids_from_actions_frame(data)
        omit_actions = []
        for (n, group) in data_with_urls.groupby('url'):
            if len(group.index) > 1:
                omit_actions += [group[group['action'] == 'page']['actionID'].tolist()[0]]

        new_data = data[~data['actionID'].isin(omit_actions)]

        assert (len(data[data['action'] == 'page']) == ((len(new_data[new_data['action'] == 'page'])) + (
            len(new_data[new_data['action'] == 'query'])))) \
               or (self._get_urls_from_frame(data).count(self.viewmystuff_url) > 1)
        return new_data

    # TODO: check state update
    def _init_new_query_segment(self, data, current_state, previous_state, total_state, queryID):
        current_state['queryID'] = queryID
        current_state['query'] = self.query_data[queryID]['query']
        current_state['bad_query_segment'] = bool(queryID in self.invalid_queryIDs)
        current_state['task_page'] = False
        current_state['num_clicks'] = 0
        current_state['pagedata_segment'] = dict()
        total_state['query_lengths'] += [len(current_state['query'].split())]
        total_state['num_queries'] += bool(queryID not in self.invalid_queryIDs)


    # Done
    # Assumes: queryIDs contains all and only the new queries issued by a user (no duplicate entries from tab revisits
    def _assign_queryid(self, data, current_state, previous_state, total_state, queryIDs):
        query_ids = []
        for d in data:
            query_ids += [row['value'] for (n, row) in d.iterrows() if row['action'] in ['query']]
            # ids += [row['value'] for (n, row) in d.iterrows() if row['action'] in ['page', 'query']]
        for val in query_ids:
            if val in queryIDs:
                # if self.intent_queryids.get(val, None) != None:
                self._init_new_query_segment(data, current_state, previous_state, total_state, val)
                break

    # Done
    # Input: Dataframe of page and query actions
    # Output: assertion of whether the frame  contains information about a single page. Permits odd cases where the Coagmento
    # viewMyStuff page was open in another window
    def _assert_singlepage_pqframe(self, data):
        assert len(set(self._get_urls_from_frame(data))) == len(data[data['action'] == 'page']) or (
            self._get_urls_from_frame(data).count(self.viewmystuff_url) > 1)


    #Done
    def _get_urls_from_grouped_data(self,data):
        urls = []
        for d in data:
            for (n,row) in d.iterrows():
                if row['action']=='query':
                    urls += [self.queryid_to_url[row['value']]]
                elif row['action']=='page':
                    urls += [self.pageid_to_url[row['value']]]
        return urls

    #Done
    def _is_new_current_query_segment(self,data,current_state,previous_state):
        return current_state['queryID'] != previous_state['queryID']

    # TODO
    def _update_pagedata(self, current_state, previous_state, total_state, action, new_query):
        try:
            elapsedtime = 0
            if previous_state.get('localTimestamp', None) is not None:
                elapsedtime = current_state['localTimestamp'] - previous_state['localTimestamp']
            elapsedtime /= 1000.0  # ms to seconds

            current_state['error_long_read'] = bool(elapsedtime >= self.long_read_errortime_seconds)
            if new_query:
                current_state['error_long_read'] = bool(elapsedtime >= self.long_queryread_errortime_seconds)

            # not a long page view.  Just misbehaving machine/browser.  Had a 10 minute wait once
            # if elapsedtime / 1000.0 >= self.long_read_errortime_seconds:
            #     raise ValueError("Long read: %f seconds, current state and action: %s %s" % (
            #         elapsedtime / 1000.0, current_state, action))
            #     return

            longpageread = elapsedtime > self.long_page_read_seconds

            # Objects to use
            #
            # current_state['tabs']
            # previous_state['tabs']
            #
            # Objects to update:
            #
            #
            # total_state['pagedata_total'] = dict()
            # current_state['pagedata_segment'] = dict()
            #
            #

            defaultdata = {
                'time_dwellfirst': 0.0,
                'time_dwelltotal': 0.0,  #
                'time_opentotal': 0.0,  #
                'views_total': 0,  #
                'firstview': False,
                'longpageread_count': 0
            }

            def f():
                try:
                    # TODO: KeyErrors
                    # print("F!")
                    # exit()
                    tabs = current_state['tabs']
                    for t in tabs:
                        if total_state['pagedata_total'].get(t, None) is None:
                            total_state['pagedata_total'][t] = copy.deepcopy(defaultdata)
                        if current_state['pagedata_segment'].get(t, None) is None:
                            current_state['pagedata_segment'][t] = copy.deepcopy(defaultdata)

                    tabs = previous_state['tabs']
                    for t in tabs:
                        if total_state['pagedata_total'].get(t, None) is None:
                            total_state['pagedata_total'][t] = copy.deepcopy(defaultdata)
                        if current_state['pagedata_segment'].get(t, None) is None:
                            current_state['pagedata_segment'][t] = copy.deepcopy(defaultdata)

                    tabs = previous_state['tabs']
                    tabIndex = previous_state['tabIndex']
                    # TODO: Check for correctness
                    for (t, i) in zip(tabs, range(len(tabs))):
                        if (not previous_state['task_page']) and (not current_state['error_long_read']):
                            total_state['pagedata_total'][t]['time_opentotal'] += elapsedtime
                            current_state['pagedata_segment'][t]['time_opentotal'] += elapsedtime

                            total_state['pagedata_total'][t]['longpageread_count'] += int(longpageread)
                            current_state['pagedata_segment'][t]['longpageread_count'] += int(longpageread)

                            if i == tabIndex:
                                total_state['pagedata_total'][t]['time_dwelltotal'] += elapsedtime
                                current_state['pagedata_segment'][t]['time_dwelltotal'] += elapsedtime

                    tabs = current_state['tabs']
                    for (t, i) in zip(tabs, range(len(tabs))):
                        if (not current_state['task_page']) and (not current_state['error_long_read']):
                            if i == tabIndex:
                                total_state['pagedata_total'][t]['views_total'] += 1
                                current_state['pagedata_segment'][t]['views_total'] += 1
                    # for (t, i) in zip(tabs, range(len(tabs))):
                    #     if (not previous_state['task_page']) and (not current_state['error_long_read']):
                    #         # print(t)
                    #         # print("IN TOTAL?",t in total_state['pagedata_total'].keys())
                    #         # print(total_state['pagedata_total'].keys())
                    #         # print("IN CURRENT?", t in current_state['pagedata_segment'].keys())
                    #         # print(current_state['pagedata_segment'].keys())
                    #         total_state['pagedata_total'][t]['time_opentotal'] += elapsedtime
                    #         current_state['pagedata_segment'][t]['time_opentotal'] += elapsedtime
                    #
                    #         total_state['pagedata_total'][t]['longpageread_count'] += int(longpageread)
                    #         current_state['pagedata_segment'][t]['longpageread_count'] += int(longpageread)
                    #
                    #         if i == tabIndex:
                    #             total_state['pagedata_total'][t]['views_total'] += 1
                    #             current_state['pagedata_segment'][t]['views_total'] += 1
                    #             total_state['pagedata_total'][t]['time_dwelltotal'] += elapsedtime
                    #             current_state['pagedata_segment'][t]['time_dwelltotal'] += elapsedtime

                except Exception as ex:
                    print("DEF F() EXCEPTION")
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    print(exc_type, exc_obj, exc_tb)
                    print("%s, line %d" % (exc_type.__name__, exc_tb.tb_lineno))
                    print(tabs)
                    print(tabIndex)
                    print(total_state['pagedata_total'])
                    print(current_state['pagedata_segment'])

                    exit()

            # if action == 'pq:noaction':
            #     # same as above
            #     f()
            #     pass
            # el
            if action == 'pq:selectchange':
                # use previous state to compute times
                # add new pages to pages tab
                f()
                pass
            else:
                print("ACTION:", action)
                exit()
        except Exception as ex:
            print("UPDATE PAGEDATA EXCEPTION")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_obj, exc_tb)
            print("%s, line %d" % (exc_type.__name__, exc_tb.tb_lineno))
            print(current_state)
            print(previous_state)
            exit()

    # TODO
    # 2) before page: possibly tabAdded+tabSelected+page. tabSelected should be combined with previous tabAdded if tabAdded is small
    # 3) before page+query: tab change, if within a third of a second.  Verify and assert.  Otherwise something is wrong.
    # 4) before page, page+query: usually within 20 ms.
    # 7) after page/page+query, within a few ms:
    def _pagequery_behavior(self, data, prev_data, current_state, previous_state, total_state, queryIDs):
        current_state['pq_prev_data'] = previous_state.get('pq_current_data', None)
        current_state['pq_current_data'] = data
        current_state['pq_prev_localTimestamp'] = previous_state.get('pq_current_localTimestamp', 0)
        current_state['pq_current_localTimestamp'] = current_state['localTimestamp']
        current_state['pq_prev_action_type'] = previous_state.get('pq_current_action_type', None)


        for d in data:
            self._assert_singlepage_pqframe(d)
        self._assign_queryid(data, current_state, previous_state, total_state, queryIDs)
        data = [self._clean_duplicates_from_pqframe(d) for d in data]

        curr_action = ['+'.join(d['action'].tolist()) for d in data]
        curr_action = '+'.join(curr_action)
        curr_action = '+'.join(sorted(curr_action.split('+')))
        current_state['pq_current_action_type'] = curr_action
        current_state['pq_current_action_count'] += 1

        # Notes:
        # 1) Account for tab with unknown url (i.e. '')
        # 2) Account for changing tabs to maintask.php, which may not be present in tabs
        try:
            pageupdate_action = ''
            new_query = False
            if curr_action.count('+') == curr_action.count('page') + curr_action.count('query') - 1:
                # TODO: Could be a delayed load.  If page already exists, ignore?
                # TODO: Handle case where just a bunch of viewmytask and current tab interleaved. See actionID=177400 for details

                # Only page/query actions ('page','query','page+page','page+query',...)
                # TODO: Count=1 current tab has new page

                # TODO: Notes for 2 page/query actions---
                # TODO: put all url's together and see if overlap.  If completely disjoint, then just select a tab.
                # TODO: Otherwise,  a) replace blank tab with new url b) replace current tab, if blank is absent

                # TODO: Other notes
                # TODO: In case of + = 3, pq = 4, there was a new window in the current tab, and it was worth editing the current tab (edit was a new query)
                # TODO: In case of + = 4, pq = 5, a person pressed back button
                # TODO: In case of + = 5, pq = 6, another person pressed back button
                # TODO: In case of + = 7, pq = 8, a person pressed back button
                #
                #
                # TODO: Account for clicking "view my stuff" button.  Doesn't do a tabAdded or tabSelected action
                # TODO: clicking another window just does a pq action, not a tabSelected action
                # TODO: Make sure that hashtag google.com/search...#... case is properly handled
                #
                # Note: they could be typing in a new query.  So in current tab, but page change

                if len(current_state['tabs']) == 0:
                    current_state['tabs'] = ['']
                    current_state['tabIndex'] = 0

                # TODO: Theory: multi is on a click or a back button, not selecting new window.  So no need to switch.  Wait for load
                if curr_action in ['page',
                                   'query'] or not self.viewmystuff_url in \
                                                   current_state['tabs']:
                    # TODO: Checkpoint
                    vals = [v for d in data for v in d['value'].tolist()]
                    # TODO: not the same!!!

                    urls = self._get_urls_from_grouped_data(data)
                    if self._is_new_current_query_segment(data,current_state,previous_state):
                        # TODO:
                        # 1) Extract url for the new query segment.
                        # 2) Assign this to current tab
                        url = self.query_data[current_state['queryID']]['url']
                        current_state['tabs'][current_state['tabIndex']] = url
                        pageupdate_action = 'pq:selectchange'
                        new_query = True
                    else:
                        ordered_disjoint_urls = [u for u in urls if not u in current_state['tabs']]
                        if len(ordered_disjoint_urls) == 0:
                            # TODO: Check
                            # TODO:
                            # 1) If current tab is viewMyStuff, pick the first non viewMyStuff tab
                            # 2) If the current tab is not viewMyStuff, pick the first viewMyStuff tab
                            #
                            # (Alt) Assumes this is just a tabSelect.  Just pick the first one.  Good choice?
                            url = urls[0]
                            selected_tab_index = current_state['tabs'].index(url)
                            if url != '':
                                current_state['tabIndex'] = selected_tab_index
                            current_state['task_page'] = bool((self.substring_coagmento in url) or (self.substring_etherpad in url) or (sum([sub in url for sub in self.substrings_badurls])>0))
                            pageupdate_action = 'pq:selectchange'
                        else:
                            url = ordered_disjoint_urls[0]

                            if url == self.viewmystuff_url:
                                if url in current_state['tabs']:
                                    current_state['tabIndex'] = current_state['tabs'].index(url)
                                    current_state['task_page'] = bool(
                                        (self.substring_coagmento in url) or (self.substring_etherpad in url) or (
                                                    sum([sub in url for sub in self.substrings_badurls]) > 0))
                                    pageupdate_action = 'pq:selectchange'
                                else:
                                    current_state['tabs'] += [url]
                                    current_state['tabIndex'] = len(current_state['tabs']) - 1
                                    current_state['task_page'] = bool(
                                        (self.substring_coagmento in url) or (self.substring_etherpad in url) or (
                                                    sum([sub in url for sub in self.substrings_badurls]) > 0))
                                    pageupdate_action = 'pq:addselect'
                            else:
                                current_state['tabs'][current_state['tabIndex']] = url
                                current_state['task_page'] = bool(
                                    (self.substring_coagmento in url) or (self.substring_etherpad in url) or (
                                                sum([sub in url for sub in self.substrings_badurls]) > 0))
                                pageupdate_action = 'pq:selectchange'
            else:
                print("ERROR!")
                print(curr_action)
                exit()
                # TODO: What is this?
                if self.missing_action_counts.get(curr_action, None) is None:
                    self.missing_action_counts[curr_action] = 1
                else:
                    self.missing_action_counts[curr_action] += 1
            self._update_pagedata(current_state, previous_state, total_state, pageupdate_action,new_query)
        except SystemExit:
            exit()
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            current_state['pq_error'] = "%s, line %d,%s" % (exc_type.__name__, exc_tb.tb_lineno,ex.message)

    # TODO
    def _create_feature_vector(self, current_state, previous_state, total_state):
        f_vec = dict()
        f_vec['userID'] = total_state['userID']
        f_vec['queryID'] = current_state['queryID']
        f_vec['bad_query_segment'] = current_state['bad_query_segment']
        f_vec['task_page'] = current_state['task_page']
        f_vec['action_type'] = current_state['action_type']


        for survey_input in self.questionnaire_columns_presearch+self.questionnaire_columns_postsearch:
            f_vec[survey_input] = total_state[survey_input]




        # for debug purposes
        # print: 1) action type 2) tabs 3) current tab
        f_vec['pq_current_action_type'] = current_state.get('pq_current_action_type', '')
        f_vec['pq_current_action_count'] = current_state.get('pq_current_action_count', 0)
        f_vec['pq_error'] = current_state.get('pq_error', '')
        f_vec['tabs'] = str([t for t in current_state['tabs']])
        if current_state['tabIndex'] is not None and current_state['tabIndex'] < len(current_state['tabs']):
            f_vec['current_tab'] = current_state['tabs'][current_state['tabIndex']]
        else:
            f_vec['current_tab'] = "ERROR"

        f_vec['questionID'] = total_state['questionID']
        f_vec['task_topic'] = total_state['task_topic']
        f_vec['task_type'] = total_state['task_type']
        f_vec['facet_product'] = total_state['facet_product']
        f_vec['facet_goal'] = total_state['facet_goal']

        #TODO: Facet val for topics (binary variable)

        f_vec['topic_familiarity'] = total_state['topic_knowledge']
        f_vec['search_years'] = total_state['searchexperience_years']
        f_vec['search_frequency'] = total_state['search_perday']
        f_vec['search_expertise'] = total_state['search_skills']
        f_vec['task_difficulty'] = total_state['task_difficulty']

        f_vec['localTimestamp'] = total_state['localTimestamp']


        f_vec['time_total'] = total_state['total_time']

        f_vec['query_length'] = len(current_state['query'].split())
        f_vec['queries_num'] = total_state['num_queries']

        def get_page_type(pt, segmentorsession):
            related_data = None
            if segmentorsession == 'segment':
                related_data = current_state['pagedata_segment']
            else:
                related_data = total_state['pagedata_total']

            related_data = dict([(k, v) for (k, v) in related_data.iteritems() if k != ''])
            # returns dictionary of page information.  Keys are url's, values are times
            if pt == 'serp':
                return dict([(p, v) for (p, v) in related_data.iteritems() if 'google.com/search' in p])
            elif pt == 'content':
                return dict([(p, v) for (p, v) in related_data.iteritems() if 'google.com/search' not in p])
            else:
                print("get_page_type alternate type", pt)
                exit()

        for seg_session in ['segment', 'session']:
            for pagetype in ['serp', 'content']:
                # for pagetype in ['serp', 'saved', 'unsaved', 'notsaved', 'content']:
                related_pages = get_page_type(pagetype, seg_session)
                for valuetype in ['time_dwelltotal']:
                # for valuetype in ['time_dwelltotal', 'time_opentotal']:
                    values_list = [values[valuetype] for (p, values) in related_pages.iteritems()]
                    if len(values_list) == 0:
                        f_vec["%s_total_%s_%s" % (valuetype, pagetype, seg_session)] = 0.0
                    else:
                        f_vec["%s_total_%s_%s" % (valuetype, pagetype, seg_session)] = np.sum(
                            [values[valuetype] for (p, values) in related_pages.iteritems()])

        segment_urls = set([url for url in current_state['pagedata_segment'].keys() if
                            url not in ['about:blank', 'about:newtab', '']])
        f_vec['pages_num_segment'] = sum(
            [((current_state['pagedata_segment'][url]['views_total'] > 0) and ('google.com/webhp' not in url) and (
                        'google.com/search' not in url)) for url in segment_urls])

        session_urls = set(
            [url for url in total_state['pagedata_total'].keys() if url not in ['about:blank', 'about:newtab', '']])
        f_vec['pages_num_session'] = sum(
            [((total_state['pagedata_total'][url]['views_total'] > 0) and ('google.com/webhp' not in url) and (
                        'google.com/search' not in url)) for url in session_urls])

        # TODO: total time elapsed?
        return f_vec

    # TODO
    def _update_total_state(self, current_state, previous_state, total_state):
        # Metadata should be constant
        total_state['localTimestamp'] = current_state['localTimestamp']
        total_state['total_time'] = total_state['localTimestamp'] - total_state['startTimestamp']
        total_state['percent_time_relative'] = float(total_state['total_time']) / float(
            total_state['endTimestamp'] - total_state['startTimestamp'])
        total_state['percent_time_20mins'] = float(total_state['total_time']) / 1200000.0

    # TODO
    def _init_states(self, current_state, total_state, previous_state):
        current_state['num_clicks'] = 0
        current_state['query'] = None
        current_state['localTimestamp'] = None
        current_state['tabs'] = []
        current_state['tabIndex'] = None
        current_state['error_long_read'] = False

        previous_state['num_clicks'] = 0
        previous_state['query'] = None
        previous_state['localTimestamp'] = None
        previous_state['tabs'] = []
        previous_state['tabIndex'] = None

        current_state['queryID'] = None
        current_state['pq_current_action_count'] = 0
        current_state['pagedata_segment'] = dict()

        total_state['num_clicks'] = 0
        total_state['num_queries'] = 0
        total_state['query_lengths'] = []
        total_state['queries'] = []
        total_state['pagedata_total'] = dict()

    # Done
    # Creates ordered list of feature vectors representing an ordered list of user's states
    # Input: A list of the user's actions (with 'localTimestamp' column), session metadata, queryIDs of issued queries
    # Output: Ordered sequence of states (ordered by localTimestamp)
    def _create_feature_sequence(self, actions_df, session_metadata, queryIDs):
        total_state = copy.deepcopy(session_metadata)
        previous_state = copy.deepcopy(session_metadata)
        current_state = copy.deepcopy(session_metadata)
        feature_vectors = []

        # add default values to total, previous, current variables
        self._init_states(current_state, total_state, previous_state)

        # Step 1) Handle non-pq actions
        groups_list = []
        for (localTimestamp, group) in actions_df.groupby('localTimestamp'):
            action = group['action'].tolist()
            action = tuple(sorted(action))
            if self.action_map.get(action, None) != self._pagequery_behavior:
                groups_list += [([group], localTimestamp)]


        # Step 2) Handle pq groups.  frames are grouped together if their timestamps are within a threshold
        group = None
        localTimestamp = None
        current_group = []
        for (n, group) in actions_df.groupby('localTimestamp'):
            localTimestamp = n
            action = group['action'].tolist()
            action = tuple(sorted(action))
            if self.action_map.get(action, None) == self._pagequery_behavior:
                assert action != ('left-click',)
                if current_group == []:
                    current_group = [group]
                else:
                    previousLocalTimestamp = current_group[-1]['localTimestamp'].tolist()[0]
                    if localTimestamp - previousLocalTimestamp < self.same_action_elapsed_time:
                        current_group += [group]
                    else:
                        minTimestamp = min([g['localTimestamp'].tolist()[0] for g in current_group])
                        groups_list += [(current_group, minTimestamp)]
                        current_group = [group]
        if current_group == []:
            groups_list += [([group], localTimestamp)]
        else:
            groups_list += [(current_group, localTimestamp)]

        # Step 3) Sort groups by timestamp
        groups_list = sorted(groups_list, key=lambda x: x[1])

        # Step 4) Stepwise processing
        prev_group = None
        for (group, localTimestamp) in groups_list:
            # Overwrite previous state
            previous_state = copy.deepcopy(current_state)
            current_state['localTimestamp'] = localTimestamp
            current_state['action_type'] = ''

            primary_group = group[-1]
            action = primary_group['action'].tolist()
            action = tuple(sorted(action))
            current_state['action_type'] = '+'.join(action)
            if self.action_map.get(action, None) == None:
                raise ValueError("Unexpected value for action and primary group: %s %s" % (action,primary_group))
            else:
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                if type(self.action_map.get(action, None)) != bool:
                    if self.action_map[action] != self._pagequery_behavior:
                        # Never reached in this data
                        self.action_map[action](group[0], prev_group, current_state, previous_state, total_state)
                    else:
                        self.action_map[action](group, prev_group, current_state, previous_state, total_state,
                                                queryIDs)

                self._update_total_state(current_state, previous_state, total_state)
                if self.action_map.get(action, None) is not None and self.action_map[action] != self._pagequery_behavior:
                    raise ValueError("Unexpected value for action: %s"%(action))
                    self._update_pagedata(current_state, previous_state, total_state, action,False)

                feature_vectors += [self._create_feature_vector(current_state, previous_state, total_state)]
            prev_group = group

        # Step 5) Output
        return pd.DataFrame(feature_vectors)

    # Done
    # Given: metadata on session (user demographics, pretask, posttask
    # Output: pandas DataFrame, where each row is a query segment and each column is a statistic
    @action_states_for_user_task_decorator
    def _process_session_data(self, session_metadata):
        userID = session_metadata['userID']
        task_num = session_metadata['task_num']
        end_timestamp = session_metadata['endTimestamp']  # millis
        stageID_task = self.tasknum_to_taskstageid[task_num]

        # TODO: More accurate extraction of queryID's
        # TODO: Account for bad queryID's?
        print("USERSTAGE",userID,stageID_task)
        queryIDs = self._extract_issued_queryIDs(userID,stageID_task)
        print("QUERYIDS",queryIDs)
        first_queryID = queryIDs[0]

        # Clean actions log for sequential processing
        actions_df = self._sql_query(
            "SELECT * from actions WHERE userID=%d AND stageID=%d AND `action` IN %s ORDER BY actionID ASC" % (
                userID, stageID_task, str(tuple(self.valid_actions))))
        # value -> numeric
        actions_df['value'] = actions_df['value'].apply(pd.to_numeric)
        # Assign localTimestamp of timestamp*1000 (conversion to milliseconds) for those with localTimestamp of zero
        actions_df.ix[actions_df.localTimestamp == 0, 'localTimestamp'] = actions_df.ix[actions_df.localTimestamp == 0, 'timestamp'] * 1000.0
        assert len(actions_df[actions_df['localTimestamp'] == 0]) == 0
        first_query_actionID = \
            actions_df[(actions_df['value'] == first_queryID) & actions_df['action'].isin(self.actions_query)][
                'actionID'].tolist()[0]
        first_query_localTimestamp = actions_df[actions_df['actionID']==first_query_actionID]['localTimestamp'].tolist()[0]
        # Filter everything between start and stop time
        actions_df = actions_df[actions_df['localTimestamp'] >= first_query_localTimestamp]
        actions_df = actions_df[actions_df['timestamp'] <= end_timestamp]
        actions_df['action'] = actions_df['action'].map(lambda x: x if x not in self.actions_query_transform else 'query')

        return self._create_feature_sequence(actions_df, session_metadata, queryIDs)

    # Done
    def process_log(self):
        # Create data frame (full_data_frame) where each row represents a query segment
        self._assert_valid_instance()

        # Create a pandas DataFrame for each session, concatenate them in the end with pd.concat
        total_features = []
        for userID in self.completed_userIDs:
            for task_num in self.task_numbers:
                if (userID, task_num) in self.skip_sessions:
                    continue
                session_metadata = {'userID': userID, 'task_num': task_num}
                session_metadata.update(self._get_session_metadata(userID, task_num))  # Get metadata for the session (user demographics, pretask, posttask)
                total_features += [self._process_session_data(
                    session_metadata)]  # return session data as a pandas DataFrame (each row is a query segment/ action...in this case just query segments)

        # TODO: Rename columns, remove unnecessary columns
        self.full_data_frame = pd.concat(total_features)
        self.full_data_frame.to_csv(self.output_csv)
        self.valid = False


    # TODO
    def slice_frame(self, override=False):
        if self.auxiliary_datasets is not None and not override:
            print("Auxiliary datasets already created!")
            return self.auxiliary_datasets
        self.auxiliary_datasets = dict()
        dataset_byquery = []
        for (n, group) in self.full_data_frame.groupby(['queryID']):
            dataset_byquery += [group.tail(1)]
            print(group.tail(1))

        self.auxiliary_datasets['byquery'] = pd.concat(dataset_byquery)
        self.auxiliary_datasets['byquery']['facet_goal_val_amorphous'] = (self.auxiliary_datasets['byquery'][
                                                                             'facet_goal'] == 'Amorphous') * 1
        self.auxiliary_datasets['byquery']['facet_product_val_intellectual'] = (self.auxiliary_datasets['byquery'][
                                                                               'facet_product'] == 'Mixed') * 0.5 + (self.auxiliary_datasets['byquery']['facet_product'] == 'Intellectual') * 1.0

        self.auxiliary_datasets['byquery_cleaned'] = self.auxiliary_datasets['byquery'].copy(deep=True)
        self.auxiliary_datasets['byquery_cleaned'] = self.auxiliary_datasets['byquery_cleaned'][
            self.auxiliary_datasets['byquery_cleaned']['bad_query_segment'] == False]

        d1 = {'10 years':10,
              '10+':10,
              "As long as I've been using the internet. Probably at least 8-10 years":10,
              'Since Junior High':6,
              '8-10 years':10,
              'about 12 years':12
              }
        d2 = {
            '1-3':1,
            '4-6':2,
                 '7-10':3,
            '10+':4,
        }
        f1 = lambda x: d1.get(x,x)
        f2 = lambda x: d2.get(x, x)
        self.auxiliary_datasets['byquery_cleaned']['search_years'] = self.auxiliary_datasets['byquery_cleaned']['search_years'].apply(f1)
        self.auxiliary_datasets['byquery_cleaned']['search_frequency'] = self.auxiliary_datasets['byquery_cleaned'][
            'search_frequency'].apply(f2)
        # TODO: Clean search years data
        return self.auxiliary_datasets


# Done
if __name__ == '__main__':
    logger = SALLogParser(output_csv='/Users/Matt/Desktop/features_sal_alldata.csv')
    logger.process_log()
    subframes = logger.slice_frame()
    subframes['byquery'].to_csv('/Users/Matt/Desktop/features_sal_byquery.csv')
    subframes['byquery_cleaned'].to_csv('/Users/Matt/Desktop/features_sal_byquery_cleaned.csv')
    exit()
#
# Query Segment Data Structure
# {
#
# last_action_timestamp: int, # 'timestamp' of last action
# startTimestamp: int, #start of task
# endTimestamp: int, #end of task
# query:str, #the query
# query_length:int, # number of words
#
# action_history: [ #full list of actions
#   (action1=str,actionTimestamp1=int)
#   ...
#   ],
#
#
# pages:{
#
#   'url1':
#   {
#   'source':str,
#   'decision_time':int,
#   'open_time':int,
#   'active_time':int
#   }
#   ,
#   '':,
#   ...
#   },
#
# current_tabs: [   # current tabs - update on any page actions, query actions, tabbing actions
# {'pageID':int,
#  'url':str,
#  'active':boolean,
#   'decision_time':1,
#   'open_time':5,
#   'active_time':1
#  },
# {'pageID':11354,
#  'url':'https://nature.com',
#  'active':False,
#   'decision_time':1,
#   'open_time':5,
#   'active_time':1
# },
# # {'pageID':int,
#  'url':'https://google.com/search?...',
#  'active':True,
#   'decision_time':1,
#   'open_time':5,
#   'active_time':1
# },
#
# ...
# ],
#
#
# queryID:int, #from video_intent_assignments
# query_data:dict,
# }
#
