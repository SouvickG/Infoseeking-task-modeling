from __future__ import print_function
from urlparse import urlparse
import sys
import copy
import pprint
import random
import traceback
import pandas as pd
import numpy as np
import sys
import mysql.connector
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
        print("OUTPUT LENGTH: %d" % len(returnval))
        return returnval

    return wrapper




class Sum14LogParser:



    tasknum_to_taskstageid = {1: 70, 2: 170}
    userID_limit = 405 #ID's smaller than this are omitted


    same_action_elapsed_time = 50
    long_page_read_seconds = 17.45  # if greater than this, long page read
    click_actions = ['left-click', 'left-dblclick',
                     'middle-click', 'right-click', 'right-dblclick']
    query_actions = ['query and 1', 'query and 4']
    bookmark_save_actions = ['Save Bookmark']
    bookmark_unsave_actions = ['delete_bookmarks']
    page_actions = ['page']
    tab_actions = ['tabAdded', 'tabClosed', 'tabSelected']
    valid_actions = click_actions + query_actions + bookmark_unsave_actions + bookmark_save_actions + page_actions + tab_actions
    viewmystuff_url = u'http://coagmento.org/fall2015intent/services/viewMyStuff.php?=true'
    queryid_threshold = 10000

    stopwords = set(stopwords.words('english') + get_stop_words('en'))
    curr_prev_action = dict()
    action_lengths = dict()





    def _set_connection(self,con):
        self.connection = con

    def _sql_query(self,query):
        return pd.read_sql(query,con=self.connection)

    def __init__(self,output_csv='./output.csv'):

        #TODO: Exclude users.  Refer to Summer2014StudyConnection.py
        self.auxiliary_datasets = None
        self.action_map = {
            ('left-click',): self._left_click,  # CHECK
            ('left-dblclick',): self._left_doubleclick,  # CHECK
            ('middle-click',): self._middle_click,  # CHECK
            ('right-click',): self._right_click,  # CHECK
            ('right-dblclick',): self._right_doubleclick,  # CHECK
            (u'left-click', u'left-dblclick'): self._left_click_left_doubleclick,  # Make same click
            ('Save Bookmark',): self._save_bookmark,  # CHECK
            (u'delete_bookmarks',): self._delete_bookmark,  # CHECK

            # tabSelects
            ('tabSelected',): self._pagequery_behavior,  # In isolation? user3: tabSelected + page/query in close vicinity
            ('page', 'tabSelected'): self._pagequery_behavior,  # As you'd expect, probably
            ('page', 'query', 'tabSelected'): self._pagequery_behavior,  # As you'd expect, probably

            # tabCloses
            (u'page', u'query', 'tabClosed'): self._pagequery_behavior,
            # Confirm the page type.  Also, why no tab selected? For user39: tabClosed-page indicates the originating page when the action occurs.  Could be anything.  Should look at immediately next action to see if the current tab was closed

            # tabAdds
            ('tabAdded',): self._pagequery_behavior,
            # In isolation? user66: ctrl+click, user82: tab added in isolation, also clicking an element which created and selected a new tab, user 7 added tabs with ctrl+click, user29 added a tab with ctrl+click
            ('page', 'tabAdded'): self._pagequery_behavior,
            # Confirm the page type. Also, why no tab selected? user21: tabAdded-page.  page was a delay?  Shouldn't be there? user39 seems to confirm the same.  the tabAdded is from the originating same page.  tabSelected-page will always occur simultaneously.  Yes, the page in this action is the originating page
            (u'page', u'query', 'tabAdded'): self._pagequery_behavior,
            # Confirm the page type.  Also, why no tab selected? Is the page in this action the originating page? Yes, according to user26

            # pages+queries
            ('page',): self._pagequery_behavior,  # As you'd expect, probably, unless in close vicinity of other actions
            ('page', 'query'): self._pagequery_behavior,  # As you'd expect, probably, unless in close vicinity of other actions
            (u'page', u'page', u'query'): self._pagequery_behavior,
            # What's going on here? user80: weird time delay bug.  User 48 opened a new tab but an old tab was still being logged (thankfully previously existed in log).  user 82 had multiple windows.  Temp solution: 1) condense to as many pages as possible.  If still multiple pages: if an existing page, probably in a different window so ignore.  For the other, change current tab to new one.  Can't assume which one comes first.  Just assume the page
            (u'page', u'page', u'query', u'query'): self._pagequery_behavior,
            # What's going on here? user4: multiple windows, one is new. user3 all one window, but new page and some previously exist.  So clean those that aready exist.  If new, assume it's a new page.  Or rather, do the same behavior as with any other pages.  Another hidden tab.  Presumably no way to tell which one is first.  When in doubt, just assume current tab and assume the duplicate information is somewhere else.
            (u'page', u'page'): self._pagequery_behavior,
            # What's going on here? user82 has multiple windows open, one is in the background.  user55, also background tabs. user35 also has background tabs
            # viewMyStuff
            (u'page', u'page', u'page'): self._pagequery_behavior,  # What's going on here?  Probably same as above
            (u'page', u'page', u'page', u'page', u'query', u'query'): self._pagequery_behavior,
            # What's going on here? Probably same as above
            (u'page', u'page', u'page', u'query'): self._pagequery_behavior,  # What's going on here? Probably same as above

            # same class
            ('tabClosed',): self._pagequery_behavior,
            # In isolation? user56: closed current tab.  user55 as well. Can also be closed when not highlighted - tabClosed occurs in isolation then, with no tabSelected (tricker to debug).
            ('page', 'tabClosed'): self._pagequery_behavior,
            # Confirm the page type.  Also, why no tab selected? For user39: tabClosed-page indicates the originating page when the action occurs.  Could be anything.  Should look at immediately next action to see if the current tab was closed
        }
        self.valid = True
        self._set_connection(mysql.connector.connect(host='localhost', user='root', password='root', database='summer2014_userstudy'))


        stage_to_sessionnum = {80:1,200:2,70:1,170:2}


        self.session_to_cognitiveq = dict()
        for (n, row) in self._sql_query("SELECT * FROM questionnaire_cognitive").iterrows():
            self.session_to_cognitiveq[(row['userID'], stage_to_sessionnum[row['stageID']])] = {
                'q1_mentally': row['q1_mentally'],
                'q2_physically': row['q2_physically'],
                'q3_rushed': row['q3_rushed'],
                'q4_success': row['q4_success'],
                'q5_hard': row['q5_hard'],
                'q6_negativeAffect': row['q6_negativeAffect'],
                'q7_learning': row['q7_learning'],
                'q8_interest': row['q8_interest']
            }

        self.session_to_demographicq = dict()
        for (n, row) in self._sql_query("SELECT * FROM questionnaire_demographic").iterrows():
            self.session_to_demographicq[row['userID']] = {
                'searchExperience': row['searchExperience'],
                'oftenSearch': row['oftenSearch'],
                'mostUsedSearchEngine': row['mostUsedSearchEngine']
            }

        self.session_to_pretaskq = dict()
        for (n, row) in self._sql_query("SELECT * FROM questionnaire_pretask WHERE stageID >= 70").iterrows():
            self.session_to_posttaskq[(row['userID'], stage_to_sessionnum[row['stageID']])] = {
                'familiarity': row['familiarity'],
                'prePerceivedDifficulty': row['prePerceivedDifficulty']
            }

        self.session_to_posttaskq = dict()
        for (n, row) in self._sql_query("SELECT * FROM questionnaire_posttask WHERE stageID >= 70").iterrows():
            self.session_to_posttaskq[(row['userID'], stage_to_sessionnum[row['stageID']])] = {
                'postPerceivedDifficulty': row['postPerceivedDifficulty'],
                'confidenceResponse': row['confidenceResponse']
            }



        self.pageid_to_url = dict([(row['pageID'], row['url'])
                              for (n, row) in self._sql_query("SELECT * from pages WHERE userID >=" + str(self.userID_limit)).iterrows()])

        self.queryid_to_url = dict([(row['queryID'], row['url']) for (n, row) in
                               self._sql_query("SELECT * from queries WHERE userID userID >=" + str(self.userID_limit)).iterrows()])


        page_df = self._sql_query("SELECT * from pages WHERE userID >=" + str(self.userID_limit))
        query_df = self._sql_query("SELECT * from queries WHERE userID >=" + str(self.userID_limit))

        self.page_data = dict(
            [(row['pageID'], {'type': 'page', 'url': row['url'], 'title': row['title'], 'source': row['source']}) for
             (n, row) in page_df.iterrows()])

        self.query_data = dict()
        for (n, row) in query_df.iterrows():
            queryrow = row
            self.query_data[row['queryID']] = {'url':queryrow['url'].tolist()[0],'title':queryrow['title'].tolist()[0],'source':queryrow['source'].tolist()[0]}
            self.query_data[row['queryID']]['query'] = queryrow['query'].tolist()[0]

        self.users_df = self._sql_query(
            "SELECT * from recruits WHERE userID IN (SELECT userID FROM users WHERE userID >= " + str(self.userID_limit) + " AND participantID IS NOT NULL)")
        userID_to_participantID = self._sql_query(
            "SELECT userID,participantID FROM users WHERE userID >= " + str(self.userID_limit) + " AND participantID IS NOT NULL")


    def is_valid(self):
        return self.valid

    def _process_page(self,data, prev_data, current_state, previous_state, total_state):
        # TODO:  If I add a page and then that accumulates, to the next action, I may add multiple times
        # TODO: Why the hell are integers in my tabs?
        # TODO: Should queries from video_intent_assignments take priority?
        current_state['pq_prev_data'] = previous_state.get('pq_current_data', None)
        current_state['pq_current_data'] = data
        prev_data = current_state['pq_prev_data']

        for d in data:
            self._assert_valid_pagequery(d)
        last_data = data[-1]
        self._assign_queryid(data, current_state, previous_state, total_state)
        data = [self._clean_query_data(d) for d in data]
        last_clean_data = data[-1]

        curr_action = 'page'

        current_state['pq_prev_localTimestamp'] = previous_state.get('pq_current_localTimestamp', 0)
        current_state['pq_current_localTimestamp'] = current_state['localTimestamp']
        current_state['pq_prev_action_type'] = previous_state.get('pq_current_action_type', None)
        current_state['pq_current_action_type'] = curr_action
        current_state['pq_current_action_count'] += 1

        try:
            pageupdate_action = ''
            if curr_action in ['page']:
                vals = [v for d in data for v in d['value'].tolist()]
                urls = [self.queryid_to_url[v] if v < self.queryid_threshold else self.pageid_to_url[v]]
                url = None
                # TODO: Get URL
                url = None
                current_state['tabs'][current_state['tabIndex']] = url
                pageupdate_action = 'pq:selectchange'
            else:
                if self.curr_prev_action.get(curr_action, None) is None:
                    self.curr_prev_action[curr_action] = 1
                else:
                    self.curr_prev_action[curr_action] += 1
            self._update_pagedata_pages(current_state, previous_state, total_state, pageupdate_action)
        except SystemExit:
            exit()
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            current_state['pq_error'] = "%s, line %d" % (exc_type.__name__, exc_tb.tb_lineno)

    def _process_query(self,data, prev_data, current_state, previous_state, total_state):
        # TODO:  If I add a page and then that accumulates, to the next action, I may add multiple times
        # TODO: Why the hell are integers in my tabs?
        # TODO: Should queries from video_intent_assignments take priority?
        current_state['pq_prev_data'] = previous_state.get('pq_current_data', None)
        current_state['pq_current_data'] = data
        prev_data = current_state['pq_prev_data']

        for d in data:
            self._assert_valid_pagequery(d)
        last_data = data[-1]
        self._assign_queryid(data, current_state, previous_state, total_state)
        data = [self._clean_query_data(d) for d in data]
        last_clean_data = data[-1]

        curr_action = ['+'.join(d['action'].tolist()) for d in data]
        curr_action = '+'.join(curr_action)
        curr_action = '+'.join(sorted(curr_action.split('+')))

        current_state['pq_prev_localTimestamp'] = previous_state.get('pq_current_localTimestamp', 0)
        current_state['pq_current_localTimestamp'] = current_state['localTimestamp']

        current_state['pq_prev_action_type'] = previous_state.get('pq_current_action_type', None)
        current_state['pq_current_action_type'] = curr_action
        current_state['pq_current_action_count'] += 1

        # Notes:
        # 1) Account for tab with unknown url (i.e. '')
        # 2) Account for changing tabs to maintask.php, which may not be present in tabs

        # TODO: New queryID  may be in any 'query' or 'query+tab....', or any 'query+...' action.  Check every time

        try:
            pageupdate_action = ''

            if curr_action == 'tabAdded':
                # Case 1) right click + add new tab
                # Just add a tab.  Cannot determine the content of the new tab
                current_state['tabs'] += ['']
                pageupdate_action = 'pq:noaction'
            elif curr_action == 'tabSelected':
                # TODO: Are these always associated with clicking a bookmark?  If so, are there other regular patterns in the data I can exploit?
                # No close preceding actions.  Do nothing.
                pageupdate_action = 'pq:noaction'
            elif curr_action == 'tabClosed':
                # User closed an inactive tab.  Cannot determine which tab was closed
                pageupdate_action = 'pq:noaction'
            elif curr_action.count('+') == curr_action.count('page') + curr_action.count('query') - 1:
                # TODO: Could be a delayed load.  If page already exists, ignore?
                # TODO: Handle case where they probably clicked a bookmark.  Standalone pq, but no tabAdded action.  It opens a new window.  Unfortunately bookmark click action doesn't have localTimestamp
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
                                   'query'] or not 'http://coagmento.org/fall2015intent/services/viewMyStuff.php?=true' in \
                        current_state['tabs']:
                    vals = [v for d in data for v in d['value'].tolist()]
                    # TODO: not the same!!!
                    urls = [self.queryid_to_url[v] if v < self.queryid_threshold else self.pageid_to_url[v]]
                    url = None
                    if sum(intentqueries) > 0:
                        urls = [u for (u, is_intentquery) in zip(urls, intentqueries) if is_intentquery]
                        url = urls[0]
                        current_state['tabs'][current_state['tabIndex']] = url
                        pageupdate_action = 'pq:selectchange'
                    else:
                        ordered_disjoint_urls = [u for u in urls if not u in current_state['tabs']]
                        if len(ordered_disjoint_urls) == 0:
                            # TODO:
                            # 1) If current tab is viewMyStuff, pick the first non viewMyStuff tab
                            # 2) If the current tab is not viewMyStuff, pick the first viewMyStuff tab
                            #
                            # (Alt) Assumes this is just a tabSelect.  Just pick the first one.  Good choice?
                            url = urls[0]
                            selected_tab_index = current_state['tabs'].index(url)
                            if url != '':
                                current_state['tabIndex'] = selected_tab_index
                            pageupdate_action = 'pq:selectchange'
                        else:
                            url = ordered_disjoint_urls[0]

                            if url == 'http://coagmento.org/fall2015intent/services/viewMyStuff.php?=true':
                                if url in current_state['tabs']:
                                    current_state['tabIndex'] = current_state['tabs'].index(url)
                                    pageupdate_action = 'pq:selectchange'
                                else:
                                    current_state['tabs'] += [url]
                                    current_state['tabIndex'] = len(current_state['tabs']) - 1
                                    pageupdate_action = 'pq:addselect'
                            else:
                                current_state['tabs'][current_state['tabIndex']] = url
                                pageupdate_action = 'pq:selectchange'

                                # try:
                                #     selected_tab_index = current_state['tabs'].index('')
                                #     current_state['tabIndex'] = selected_tab_index
                                #     current_state['tabs'][selected_tab_index] = url
                                # except ValueError:
                                #
            elif curr_action.count('+') == 1 and curr_action in ['page+tabSelected', 'query+tabSelected']:
                # find the url.  Shift to url in tabs.  Assumes it exists
                urls = self._get_selecturls_from_data(data)
                url = urls[0]

                # TODO: catchall for 'maintask.php' and 'about:home' in a preexisting tab.  but catches too many cases
                if url in ['about:home', 'http://coagmento.org/spring2016intent/instruments/maintask.php',
                           'http://coagmento.org/fall2015intent/instruments/maintask.php'] and not url in current_state[
                    'tabs']:
                    current_state['tabs'] += [url]

                (current_state['tabs'], current_state['tabIndex']) = self._select_tab(current_state['tabs'],
                                                                                current_state['tabIndex'], url)
                pageupdate_action = 'pq:select'


            elif curr_action.count('+') == curr_action.count('page') + curr_action.count(
                    'query') + 1 and curr_action.count('tabAdded') == curr_action.count('tabSelected') == 1:
                # Cases, 1 page/query, 2 page/queries, 3 page/queries
                urls = self._get_selecturls_from_data(data)
                # TODO: Just about:newtab?  Or about:newtab and about:blank?
                if not 'about:newtab' in urls:
                    print("ASSERT URLS", urls)
                assert 'about:newtab' or 'about:blank' in urls
                if 'about:newtab' in urls:
                    current_state['tabs'] += ['about:newtab']
                else:
                    current_state['tabs'] += ['about:blank']
                current_state['tabIndex'] = len(current_state['tabs']) - 1
                pageupdate_action = 'pq:addselect'

            elif curr_action.count('+') == curr_action.count('page') + curr_action.count('query') and curr_action.count(
                    'tabAdded') == 1:
                # tabAdded and any number of pq actions
                # Case 1: pq count = 1: Tab added with right click "new tab".  Add a blank tab
                # Case 2: pq count = 2: Tab added with right click "new tab".  Add a blank tab
                # Case 3: pq count = 3: Tab added with right click "new tab".  Add a blank tab
                current_state['tabs'] += ['']
                pageupdate_action = 'pq:noaction'


            elif curr_action.count('+') == curr_action.count('page') + curr_action.count(
                    'query') + 1 and curr_action.count('tabClosed') == curr_action.count('tabSelected') == 1:
                # elif curr_action.count('+')==curr_action.count('page')+curr_action.count('query')+1 and curr_action.count('tabClosed')==curr_action.count('tabSelected')==1:
                # Case pq = 1: Close current tab.  Remove it from the list of url's.  page grouped with tabSelect is the one
                # Case pq = 2: Close current tab.  Remove it from list of url's
                # Case pq = 3: Close current tab. Remove it from list of url's in data.  Remaining ones are possible url's.  (probably not a "viewMyStuff" url)
                # TODO: Is the closeTab action always grouped with the tab that is closed?  In code the current tab is printed.  Since tabClosed and tabSelected are so close , we can assume they closed current tab and select was done automatically
                old_tabs = copy.copy(current_state['tabs'])
                current_state['tabs'] = self._close_tab(current_state['tabs'], current_state['tabIndex'])
                urls = self._get_selecturls_from_data(data)
                url = urls[0]

                (current_state['tabs'], current_state['tabIndex']) = self._select_tab(current_state['tabs'],
                                                                                current_state['tabIndex'], url)
                pageupdate_action = 'pq:closeselect'




            elif curr_action.count('+') == curr_action.count('page') + curr_action.count('query') and curr_action.count(
                    'tabSelected') == 1:
                # Case pq=2: Selected the page grouped with tabSelected action
                # Case pq=3: Selected the page grouped with tabSelected action
                urls = self._get_selecturls_from_data(data)
                url = urls[0]
                (current_state['tabs'], current_state['tabIndex']) = self._select_tab(current_state['tabs'],
                                                                                current_state['tabIndex'], url)
                pageupdate_action = 'pq:select'


            elif curr_action.count('+') == 1 and curr_action.count('tabClosed') == 1 and (
                    curr_action.count('page') + curr_action.count('query') == 1):
                # Tab closed.  Cannot do anything
                # Only occurs 2 times in the data
                pageupdate_action = 'pq:noaction'
            elif curr_action == 'query+tabSelected+tabSelected':
                # TODO: Just treat like a query+tabSelected: select the correct tab
                # User restored tabs and an old tab was selected
                # Occurs 1 time in the data
                urls = self._get_selecturls_from_data(data)
                url = urls[0]
                # TODO: tab select
                (current_state['tabs'], current_state['tabIndex']) = self._select_tab(current_state['tabs'],
                                                                                current_state['tabIndex'], url)
                pageupdate_action = 'pq:select'




            elif curr_action == 'page+page+page+page+page+tabAdded+tabAdded+tabClosed+tabSelected+tabSelected+tabSelected':
                # Opened bookmark on "arctic methane" tab.  Opened window that shifted to "about:blank->Cost of Arctic".  Cost of arctic dragged to main window
                # Occurs in order:
                # 1) Open bookmark action added tab while on "arctic methane"
                # 2) tabSelect to about:blank. TODO: note to self, if selected tab doesn't exist, create it?  Shouldn't normally happen
                # 3) page transition to about:blank->cost of arctic
                # 4) ignore tabClosed.  Cost of arctic dragged to main window
                # 5) Make sure cost of arctic is selected.  url: https://www.rsm.nl/about-rsm/news/detail/2992-cost-of-arctic-methane-release-could-be-size-of-global-economy-warn-experts/
                # Occurs 1 time in the data

                url = 'https://www.rsm.nl/about-rsm/news/detail/2992-cost-of-arctic-methane-release-could-be-size-of-global-economy-warn-experts/'
                current_state['tabs'] += [url]
                current_state['tabIndex'] = current_state['tabs'].index(url)
                pageupdate_action = 'pq:addselect'
            else:
                if self.curr_prev_action.get(curr_action, None) is None:
                    self.curr_prev_action[curr_action] = 1
                else:
                    self.curr_prev_action[curr_action] += 1
            self._update_pagedata(current_state, previous_state, total_state, pageupdate_action)
        except SystemExit:
            exit()
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            current_state['pq_error'] = "%s, line %d" % (exc_type.__name__, exc_tb.tb_lineno)

    # def _click(self,data, prev_data, current_state, previous_state, total_state):
    #     current_state['num_clicks'] += 1
    #     total_state['num_clicks'] += 1
    #     current_state['action_type'] = 'click'
    #
    # def _left_click(self,data, prev_data, current_state, previous_state, total_state):
    #     self._click(data, prev_data, current_state, previous_state, total_state)
    #
    # def _left_doubleclick(self,data, prev_data, current_state, previous_state, total_state):
    #     self._click(data, prev_data, current_state, previous_state, total_state)
    #
    # def _right_click(self,data, prev_data, current_state, previous_state, total_state):
    #     self._click(data, prev_data, current_state, previous_state, total_state)
    #
    # def _right_doubleclick(self,data, prev_data, current_state, previous_state, total_state):
    #     self._click(data, prev_data, current_state, previous_state, total_state)
    #
    # def _middle_click(self,data, prev_data, current_state, previous_state, total_state):
    #     self._click(data, prev_data, current_state, previous_state, total_state)
    #
    # def _left_click_left_doubleclick(self,data, prev_data, current_state, previous_state, total_state):
    #     data = data[data['action'] == 'left-dblclick']
    #     self._left_doubleclick(data, prev_data, current_state, previous_state, total_state)
    #
    # def _save_bookmark(self,data, prev_data, current_state, previous_state, total_state):
    #     bookmarkID = data['value'].tolist()[0]
    #     bookmarks = self._sql_query("SELECT * FROM bookmarks WHERE bookmarkID=%d" % bookmarkID)
    #     bookmarkURL = bookmarks['url'].tolist()[0]
    #     current_state['bookmarks'] += [bookmarkURL]
    #     total_state['bookmarks'] += [bookmarkURL]
    #     current_state['action_type'] = 'save_bookmark'
    #     current_state['bookmark_saved_currentsegment'] = int(len(current_state['bookmarks']) > 0)
    #
    # def _delete_bookmark(self,data, prev_data, current_state, previous_state, total_state):
    #     bookmarkID = data['value'].tolist()[0]
    #     bookmarks = self._sql_query("SELECT * FROM bookmarks WHERE bookmarkID=%d" % bookmarkID)
    #     bookmarkURL = bookmarks['url'].tolist()[0]
    #     if bookmarkURL in current_state['bookmarks']:
    #         current_state['bookmarks'].remove(bookmarkURL)
    #         current_state['bookmark_unsaved_currentsegment'] = int(True)
    #     total_state['bookmarks'].remove(bookmarkURL)
    #     total_state['bookmarks_unsaved'] += [bookmarkURL]
    #     current_state['bookmarks_unsaved'] += [bookmarkURL]
    #
    #     current_state['action_type'] = 'delete_bookmark'
    #     current_state['bookmark_saved_currentsegment'] = int(len(current_state['bookmarks']) > 0)
    #
    #
    # def _update_pagedata(self,current_state, previous_state, total_state, action):
    #
    #     elapsedtime = 0
    #     if previous_state.get('localTimestamp', None) is not None:
    #         elapsedtime = current_state['localTimestamp'] - previous_state['localTimestamp']
    #     elapsedtime /= 1000.0  # ms to seconds
    #
    #     # not a long page view.  Just misbehaving machine/browser.  Had a 10 minute wait once
    #     if elapsedtime / 1000.0 >= 300.0:
    #         return
    #
    #     longpageread = elapsedtime > self.long_page_read_seconds
    #
    #     # Objects to use
    #     #
    #     # current_state['tabs']
    #     # previous_state['tabs']
    #     #
    #     # current_state['bookmarks']
    #     # previous_state['bookmarks']
    #     # current_state['bookmarks_unsaved']
    #     # previous_state['bookmarks_unsaved']
    #     #
    #     #
    #     # Objects to update:
    #     #
    #     #
    #     # total_state['pagedata_total'] = dict()
    #     # current_state['pagedata_segment'] = dict()
    #     #
    #     #
    #
    #     defaultdata = {
    #         'time_dwellfirst': 0.0,
    #         'time_dwelltotal': 0.0,  #
    #         'time_dwelltotaltosave': 0.0,  #
    #         'time_opentotal': 0.0,  #
    #         'time_opentotaltosave': 0.0,  #
    #         'views_total': 0,  #
    #         'views_totaltosave': 0,  #
    #         'firstview': False,
    #         'longpageread_count': 0
    #
    #         # others:
    #     }
    #
    #     def f():
    #         tabs = current_state['tabs']
    #         for t in tabs:
    #             if total_state['pagedata_total'].get(t, None) is None:
    #                 total_state['pagedata_total'][t] = copy.deepcopy(defaultdata)
    #             if current_state['pagedata_segment'].get(t, None) is None:
    #                 current_state['pagedata_segment'][t] = copy.deepcopy(defaultdata)
    #
    #         tabs = previous_state['tabs']
    #         tabIndex = previous_state['tabIndex']
    #         bookmarks_lookup = dict([(b, []) for b in total_state['bookmarks']])
    #         for (t, i) in zip(tabs, range(len(tabs))):
    #             total_state['pagedata_total'][t]['time_opentotal'] += elapsedtime
    #             current_state['pagedata_segment'][t]['time_opentotal'] += elapsedtime
    #
    #             total_state['pagedata_total'][t]['longpageread_count'] += int(longpageread)
    #             current_state['pagedata_segment'][t]['longpageread_count'] += int(longpageread)
    #
    #             if bookmarks_lookup.get(t, None) is None:
    #                 total_state['pagedata_total'][t]['time_opentotaltosave'] += elapsedtime
    #                 current_state['pagedata_segment'][t]['time_opentotaltosave'] += elapsedtime
    #
    #             if i == tabIndex:
    #                 total_state['pagedata_total'][t]['views_total'] += 1
    #                 current_state['pagedata_segment'][t]['views_total'] += 1
    #                 total_state['pagedata_total'][t]['time_dwelltotal'] += elapsedtime
    #                 current_state['pagedata_segment'][t]['time_dwelltotal'] += elapsedtime
    #                 if bookmarks_lookup.get(t, None) is None:
    #                     total_state['pagedata_total'][t]['views_totaltosave'] += 1
    #                     current_state['pagedata_segment'][t]['views_totaltosave'] += 1
    #                     total_state['pagedata_total'][t]['time_dwelltotaltosave'] += elapsedtime
    #                     current_state['pagedata_segment'][t]['time_dwelltotaltosave'] += elapsedtime
    #
    #     # change firstview on page-related actions, not on click+bookmark actions
    #     if type(action) == tuple and (
    #                         'left-click' in action or 'right-click' in action or 'middle-click' in action or 'left-dblclick' in action or 'right-dblclick' in action):
    #         # normal update to times
    #         f()
    #
    #     elif type(action) == tuple and ('Save Bookmark' in action or 'delete_bookmarks' in action):
    #
    #         # same as above
    #         f()
    #         pass
    #
    #     elif action == 'pq:noaction':
    #         # same as above
    #         f()
    #         pass
    #     elif action == 'pq:closeselect':
    #         f()
    #         pass
    #     elif action == 'pq:addselect':
    #         # use previous state to compute times
    #         f()
    #         pass
    #     elif action == 'pq:select':
    #         # use previous state to compute times
    #         f()
    #         pass
    #     elif action == 'pq:selectchange':
    #         # use previous state to compute times
    #         # add new pages to pages tab
    #         f()
    #         pass
    #     else:
    #         print("ACTION:", action)
    #         # exit()

    def _update_pagedata_pages(self,current_state, previous_state, total_state, action):

        elapsedtime = 0
        if previous_state.get('localTimestamp', None) is not None:
            elapsedtime = current_state['localTimestamp'] - previous_state['localTimestamp']
        elapsedtime /= 1000.0  # ms to seconds

        # not a long page view.  Just misbehaving machine/browser.  Had a 10 minute wait once
        if elapsedtime / 1000.0 >= 300.0:
            return

        longpageread = elapsedtime > self.long_page_read_seconds

        # Objects to use
        #
        # current_state['tabs']
        # previous_state['tabs']
        #
        # current_state['bookmarks']
        # previous_state['bookmarks']
        # current_state['bookmarks_unsaved']
        # previous_state['bookmarks_unsaved']
        #
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
            'time_dwelltotaltosave': 0.0,  #
            'time_opentotal': 0.0,  #
            'time_opentotaltosave': 0.0,  #
            'views_total': 0,  #
            'views_totaltosave': 0,  #
            'firstview': False,
            'longpageread_count': 0

            # others:
        }

        def f():
            tabs = current_state['tabs']
            for t in tabs:
                if total_state['pagedata_total'].get(t, None) is None:
                    total_state['pagedata_total'][t] = copy.deepcopy(defaultdata)
                if current_state['pagedata_segment'].get(t, None) is None:
                    current_state['pagedata_segment'][t] = copy.deepcopy(defaultdata)

            tabs = previous_state['tabs']
            tabIndex = previous_state['tabIndex']
            bookmarks_lookup = dict([(b, []) for b in total_state['bookmarks']])
            for (t, i) in zip(tabs, range(len(tabs))):
                total_state['pagedata_total'][t]['time_opentotal'] += elapsedtime
                current_state['pagedata_segment'][t]['time_opentotal'] += elapsedtime

                total_state['pagedata_total'][t]['longpageread_count'] += int(longpageread)
                current_state['pagedata_segment'][t]['longpageread_count'] += int(longpageread)

                if bookmarks_lookup.get(t, None) is None:
                    total_state['pagedata_total'][t]['time_opentotaltosave'] += elapsedtime
                    current_state['pagedata_segment'][t]['time_opentotaltosave'] += elapsedtime

                if i == tabIndex:
                    total_state['pagedata_total'][t]['views_total'] += 1
                    current_state['pagedata_segment'][t]['views_total'] += 1
                    total_state['pagedata_total'][t]['time_dwelltotal'] += elapsedtime
                    current_state['pagedata_segment'][t]['time_dwelltotal'] += elapsedtime
                    if bookmarks_lookup.get(t, None) is None:
                        total_state['pagedata_total'][t]['views_totaltosave'] += 1
                        current_state['pagedata_segment'][t]['views_totaltosave'] += 1
                        total_state['pagedata_total'][t]['time_dwelltotaltosave'] += elapsedtime
                        current_state['pagedata_segment'][t]['time_dwelltotaltosave'] += elapsedtime

        # change firstview on page-related actions, not on click+bookmark actions
        if type(action) == tuple and (
                            'left-click' in action or 'right-click' in action or 'middle-click' in action or 'left-dblclick' in action or 'right-dblclick' in action):
            # normal update to times
            f()

        elif type(action) == tuple and ('Save Bookmark' in action or 'delete_bookmarks' in action):

            # same as above
            f()
            pass

        elif action == 'pq:noaction':
            # same as above
            f()
            pass
        elif action == 'pq:closeselect':
            f()
            pass
        elif action == 'pq:addselect':
            # use previous state to compute times
            f()
            pass
        elif action == 'pq:select':
            # use previous state to compute times
            f()
            pass
        elif action == 'pq:selectchange':
            # use previous state to compute times
            # add new pages to pages tab
            f()
            pass
        else:
            print("ACTION:", action)
            # exit()

    # Creates ordered list of feature vectors representing an ordered list of user's states
    # Input: A list of the user's actions (with 'localTimestamp' column)
    # Output: Ordered sequence of states (ordered by localTimestamp)
    def _create_feature_sequence(self, actions_df, task_metadata):

        # TODO: The only real query segments are those marked in the intentions-table.  Can be queryID's or pageID's
        total_state = task_metadata.copy()
        previous_state = task_metadata.copy()
        current_state = task_metadata.copy()
        features = []

        # add default values to total, previous, current variabesl
        self._init_states(current_state, total_state, previous_state)

        # Order and iterate by localTimestamp
        # Loop for each action

        # non-pq actions are their own group
        groups_list = []
        for (n, group) in actions_df.groupby('localTimestamp'):
            localTimestamp = n
            action = group['action'].tolist()
            action = tuple(sorted(action))
            if action == ('left-click',):
                actionID = int(group['actionID'].tolist()[0])
            if self.action_map[action] != self._pagequery_behavior:
                groups_list += [([group], localTimestamp)]

        # pq actions make separate groups,  frames are grouped together if their timestamps are within a threshold
        group = None
        localTimestamp = None
        current_group = []
        for (n, group) in actions_df.groupby('localTimestamp'):
            localTimestamp = n
            action = group['action'].tolist()
            action = tuple(sorted(action))
            if self.action_map[action] == self._pagequery_behavior:
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
        groups_list += [([group], localTimestamp)]
        groups_list = sorted(groups_list, key=lambda x: x[1])

        prev_group = None
        for (group, localTimestamp) in groups_list:
            # Overwrite previous state
            previous_state = copy.deepcopy(current_state)

            current_state['localTimestamp'] = localTimestamp
            current_state['action_type'] = ''

            primary_group = group[-1]
            action = primary_group['action'].tolist()
            action = tuple(sorted(action))
            if self.action_map.get(action, ) == None:
                print(action)
                exit()
            if type(self.action_map.get(action, None)) != bool:
                if self.action_map[action] != self._pagequery_behavior:
                    self.action_map[action](group[0], prev_group, current_state, previous_state, total_state)
                else:
                    self.action_map[action](group, prev_group, current_state, previous_state, total_state)

            if current_state.get('query', None) is None:
                print(group, localTimestamp)
                print(groups_list[0:2])
                print("QUERY!")
            self._update_total_state(current_state, previous_state, total_state)
            if self.action_map.get(action, None) is not None and self.action_map[action] != self._pagequery_behavior:
                self._update_pagedata(current_state, previous_state, total_state, action)

            features += [self._create_feature_vector(current_state, previous_state, total_state)]
            prev_group = group

        return pd.DataFrame(features)

    def _create_feature_sequence_pages(self, pages_df, task_metadata):

        # TODO: The only real query segments are those marked in the intentions-table.  Can be queryID's or pageID's
        total_state = task_metadata.copy()
        previous_state = task_metadata.copy()
        current_state = task_metadata.copy()
        features = []

        # add default values to total, previous, current variabesl
        self._init_states_pages(current_state, total_state, previous_state)

        # Order and iterate by localTimestamp
        # Loop for each action

        # TODO: Check for actions with same localTimestamp
        # non-pq actions are their own group
        # groups_list = []
        # for (n, group) in pages_df.groupby('localTimestamp'):
        #     localTimestamp = n
        #     action = group['action'].tolist()
        #     action = tuple(sorted(action))
        #     if action == ('left-click',):
        #         actionID = int(group['actionID'].tolist()[0])
        #     if self.action_map[action] != self._pagequery_behavior:
        #         groups_list += [([group], localTimestamp)]
        #
        # # pq actions make separate groups,  frames are grouped together if their timestamps are within a threshold
        # group = None
        # localTimestamp = None
        # current_group = []
        # for (n, group) in actions_df.groupby('localTimestamp'):
        #     localTimestamp = n
        #     action = group['action'].tolist()
        #     action = tuple(sorted(action))
        #     if self.action_map[action] == self._pagequery_behavior:
        #         assert action != ('left-click',)
        #
        #         if current_group == []:
        #             current_group = [group]
        #         else:
        #             previousLocalTimestamp = current_group[-1]['localTimestamp'].tolist()[0]
        #             if localTimestamp - previousLocalTimestamp < self.same_action_elapsed_time:
        #                 current_group += [group]
        #             else:
        #                 minTimestamp = min([g['localTimestamp'].tolist()[0] for g in current_group])
        #                 groups_list += [(current_group, minTimestamp)]
        #                 current_group = [group]
        # groups_list += [([group], localTimestamp)]
        # groups_list = sorted(groups_list, key=lambda x: x[1])

        prev_group = None
        # for (group, localTimestamp) in groups_list:
        for (n, row) in pages_df.iterrows():
            # Overwrite previous state
            previous_state = copy.deepcopy(current_state)
            current_state['localTimestamp'] = row['localTimestamp']
            # current_state['action_type'] = ''

            # primary_group = group[-1]
            # action = primary_group['action'].tolist()
            # action = tuple(sorted(action))

            # 1) Check if page or query
            # 2) Do one thing if page.  Otherwise, do the other if query
            # 3) Do one thing if page.  Otherwise, do the other if query

            if True:
                pass
            else:
                pass


            self._update_total_state(current_state, previous_state, total_state)
            self._update_pagedata_pages(current_state, previous_state, total_state, action)

            features += [self._create_feature_vector(current_state, previous_state, total_state)]
            prev_group = group

        return pd.DataFrame(features)

    @action_states_for_user_task_decorator
    def _process_session_data(self,session_metadata):
        userID = session_metadata['userID']
        task_num = session_metadata['task_num']
        stageID_task = self.tasknum_to_taskstageid[task_num]


        actions_df = None


        actions_df = self._sql_query(
            "SELECT * from actions WHERE userID=%d AND stageID=%d AND `action` IN %s ORDER BY actionID ASC" % (
                userID, stageID_task, str(tuple(self.valid_actions))))

        queries_df = self._sql_query(
            "SELECT * from queries WHERE userID=%d AND stageID=%d ORDER BY `localTimestamp` ASC"%(userID,stageID_task))

        pages_df = self._sql_query(
            "SELECT * from pages WHERE userID=%d AND stageID=%d ORDER BY `localTimestamp` ASC" % (
            userID, stageID_task))

        first_query = min(queries_df['queryID'].tolist())


        end_timestamp = self._sql_query(
            "SELECT * FROM questions_progress WHERE userID=%d AND stageID=%d" % (userID, stageID_task))['endTimestamp'].tolist()[0]


        # Filter actions log
        # Assign localTimestamp of timestamp*1000 (conversion to milliseconds) for those with localTimestamp of zero
        actions_df.ix[actions_df.localTimestamp == 0, 'localTimestamp'] = actions_df.ix[
                                                                              actions_df.localTimestamp == 0, 'timestamp'] * 1000.0
        assert len(actions_df[actions_df['localTimestamp'] == 0]) == 0
        actions_df['value'] = actions_df['value'].apply(pd.to_numeric)

        first_query_localTimestamp = \
        actions_df[(actions_df['value'] == first_query) & actions_df['action'].isin(self.query_actions)][
            'localTimestamp'].tolist()[0]

        actions_df = actions_df[actions_df['localTimestamp'] >= first_query_localTimestamp]
        # Filter everything after the task stop time
        actions_df = actions_df[actions_df['timestamp'] <= end_timestamp]
        actions_df['action'] = actions_df['action'].map(
            lambda x: x if x not in ('query and 1', 'query and 4') else 'query')

        session_metadata_2 = session_metadata.copy()

        # return self._create_feature_sequence(actions_df, session_metadata)
        return self._create_feature_sequence_pages(actions_df, session_metadata)


    def _get_task_metadata(self, userID, taskNum):
        questionIDs = self._sql_query("SELECT * FROM pages WHERE userID=%d GROUP BY questionID ORDER BY questionID ASC" % userID).tolist()
        questionID = questionIDs[taskNum]
        question_data = self._sql_query("SELECT * FROM questions_study WHERE questionID=%d" % questionID)
        questionprogress_data = self._sql_query(
            "SELECT * FROM questions_progress WHERE userID=%d AND questionID=%d" % (userID, questionID))
        question_starttime = questionprogress_data['startTimestamp'].tolist()[0] * 1000
        question_endtime = questionprogress_data['endTimestamp'].tolist()[0] * 1000

        search_experience = self.session_to_demographicq[userID]['searchExperience']
        often_search = self.session_to_demographicq[userID]['oftenSearch']

        postsearch_data = self.session_to_posttaskq[(userID,taskNum)]
        post_difficult = postsearch_data['postPerceivedDifficulty'].tolist()[0]
        post_confident = postsearch_data['confidenceResponse'].tolist()[0]

        presearch_data = self.session_to_pretaskq[(userID, taskNum)]
        pre_familiarity = presearch_data['familiarity'].tolist()[0]
        pre_difficult = presearch_data['prePerceivedDifficulty'].tolist()[0]

        cognitive_data = self.session_to_pretaskq[(userID, taskNum)]
        cog_mentally = cognitive_data['q1_mentally'].tolist()[0]
        cog_physically = cognitive_data['q2_physically'].tolist()[0]
        cog_rushed = cognitive_data['q3_rushed'].tolist()[0]
        cog_success = cognitive_data['q4_success'].tolist()[0]
        cog_hard = cognitive_data['q5_hard'].tolist()[0]
        cog_negative = cognitive_data['q6_negativeAffect'].tolist()[0]
        cog_learning = cognitive_data['q7_learning'].tolist()[0]
        cog_interest = cognitive_data['q8_interest'].tolist()[0]

        #TODO: Annotate and record facet value data
        task_topic = question_data['topicAreaID'].tolist()[0]
        num_queries_total = len(self._sql_query("SELECT * FROM queries WHERE userID=%d AND questionID=%d GROUP BY `url`" % (userID,questionID)).index)

        to_return = {'startTimestamp': question_starttime,
                     'endTimestamp': question_endtime,
                     'userID': userID,
                     'questionID': questionID,
                     'task_topic': task_topic,
                     'num_queries_total': num_queries_total,
                     'post_difficult': post_difficult,
                     'post_confident': post_confident,
                     'pre_familiarity': pre_familiarity,
                     'pre_difficult': pre_difficult,
                     'cog_mentally': cog_mentally,
                     'cog_physically': cog_physically,
                     'cog_rushed': cog_rushed,
                     'cog_success': cog_success,
                     'cog_hard': cog_hard,
                     'cog_negative': cog_negative,
                     'cog_learning': cog_learning,
                     'cog_interest': cog_interest,
                     }

        return to_return



    def process_log(self):
        if not self.valid:
            print("Not a valid object.  Please create another instance of NSFLogParser")
            return

        total_features = []
        for (n, row) in self.users_df.iterrows():
            userID = row['userID']

            for task_num in [1,2]:
                session_metadata = {'userID':userID,'task_num':task_num}
                session_metadata.update(self._get_task_metadata(userID, task_num))
                total_features += [self._process_session_data(session_metadata)]
                #TODO: Checkpoint

        self.full_data_frame = pd.concat(total_features)

        self.full_data_frame.to_csv(self.output_csv)

        self.full_data_frame[
            ['userID', 'questionID', 'queryID', 'pq_error', 'pq_current_action_count', 'pq_current_action_type', 'tabs',
             'number_tabs', 'current_tab']].to_csv('/Users/Matt/Desktop/tabs_debug.csv')
        self.valid = False

    def print_info(self,group):
        print(group[['actionID', 'action', 'value', 'timestamp', 'localTimestamp']])

    # tabClosed+page/query: pressing close, the active tab is under current view.  Still no way to tell which was closed, unless select immediately follows

    # tabSelected:
    # 1) before tabSelected+page: Action shouldn't be considered.  Just use for timestamp
    # 2) before page: possibly tabAdded+tabSelected+page. tabSelected should be combined with previous tabAdded if tabAdded is small
    # 3) before page+query: tab change, if within a third of a second.  Verify and assert.  Otherwise something is wrong.
    # 4) before page, page+query: usually within 20 ms.
    # 5) after tabAdded/tabClosed, within a few ms then same action
    # 6) after tabAdded/tabClosed, within a few ms then same action
    # 7) after page/page+query, within a few ms:
    def _pagequery_behavior(self,data, prev_data, current_state, previous_state, total_state):
        # TODO:  If I add a page and then that accumulates, to the next action, I may add multiple times
        # TODO: Why the hell are integers in my tabs?
        # TODO: Should queries from video_intent_assignments take priority?
        current_state['pq_prev_data'] = previous_state.get('pq_current_data', None)
        current_state['pq_current_data'] = data
        prev_data = current_state['pq_prev_data']

        for d in data:
            self._assert_valid_pagequery(d)
        last_data = data[-1]
        self._assign_queryid(data, current_state, previous_state, total_state)
        data = [self._clean_query_data(d) for d in data]
        last_clean_data = data[-1]

        curr_action = ['+'.join(d['action'].tolist()) for d in data]
        curr_action = '+'.join(curr_action)
        curr_action = '+'.join(sorted(curr_action.split('+')))

        current_state['pq_prev_localTimestamp'] = previous_state.get('pq_current_localTimestamp', 0)
        current_state['pq_current_localTimestamp'] = current_state['localTimestamp']

        current_state['pq_prev_action_type'] = previous_state.get('pq_current_action_type', None)
        current_state['pq_current_action_type'] = curr_action
        current_state['pq_current_action_count'] += 1

        # Notes:
        # 1) Account for tab with unknown url (i.e. '')
        # 2) Account for changing tabs to maintask.php, which may not be present in tabs

        # TODO: New queryID  may be in any 'query' or 'query+tab....', or any 'query+...' action.  Check every time

        try:
            pageupdate_action = ''

            if curr_action == 'tabAdded':
                # Case 1) right click + add new tab
                # Just add a tab.  Cannot determine the content of the new tab
                current_state['tabs'] += ['']
                pageupdate_action = 'pq:noaction'
            elif curr_action == 'tabSelected':
                # TODO: Are these always associated with clicking a bookmark?  If so, are there other regular patterns in the data I can exploit?
                # No close preceding actions.  Do nothing.
                pageupdate_action = 'pq:noaction'
            elif curr_action == 'tabClosed':
                # User closed an inactive tab.  Cannot determine which tab was closed
                pageupdate_action = 'pq:noaction'
            elif curr_action.count('+') == curr_action.count('page') + curr_action.count('query') - 1:
                # TODO: Could be a delayed load.  If page already exists, ignore?
                # TODO: Handle case where they probably clicked a bookmark.  Standalone pq, but no tabAdded action.  It opens a new window.  Unfortunately bookmark click action doesn't have localTimestamp
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
                                   'query'] or not 'http://coagmento.org/fall2015intent/services/viewMyStuff.php?=true' in \
                        current_state['tabs']:
                    vals = [v for d in data for v in d['value'].tolist()]
                    # TODO: not the same!!!
                    urls = [self.queryid_to_url[v] if v < self.queryid_threshold else self.pageid_to_url[v]]
                    url = None
                    if sum(intentqueries) > 0:
                        urls = [u for (u, is_intentquery) in zip(urls, intentqueries) if is_intentquery]
                        url = urls[0]
                        current_state['tabs'][current_state['tabIndex']] = url
                        pageupdate_action = 'pq:selectchange'
                    else:
                        ordered_disjoint_urls = [u for u in urls if not u in current_state['tabs']]
                        if len(ordered_disjoint_urls) == 0:
                            # TODO:
                            # 1) If current tab is viewMyStuff, pick the first non viewMyStuff tab
                            # 2) If the current tab is not viewMyStuff, pick the first viewMyStuff tab
                            #
                            # (Alt) Assumes this is just a tabSelect.  Just pick the first one.  Good choice?
                            url = urls[0]
                            selected_tab_index = current_state['tabs'].index(url)
                            if url != '':
                                current_state['tabIndex'] = selected_tab_index
                            pageupdate_action = 'pq:selectchange'
                        else:
                            url = ordered_disjoint_urls[0]

                            if url == 'http://coagmento.org/fall2015intent/services/viewMyStuff.php?=true':
                                if url in current_state['tabs']:
                                    current_state['tabIndex'] = current_state['tabs'].index(url)
                                    pageupdate_action = 'pq:selectchange'
                                else:
                                    current_state['tabs'] += [url]
                                    current_state['tabIndex'] = len(current_state['tabs']) - 1
                                    pageupdate_action = 'pq:addselect'
                            else:
                                current_state['tabs'][current_state['tabIndex']] = url
                                pageupdate_action = 'pq:selectchange'

                                # try:
                                #     selected_tab_index = current_state['tabs'].index('')
                                #     current_state['tabIndex'] = selected_tab_index
                                #     current_state['tabs'][selected_tab_index] = url
                                # except ValueError:
                                #
            elif curr_action.count('+') == 1 and curr_action in ['page+tabSelected', 'query+tabSelected']:
                # find the url.  Shift to url in tabs.  Assumes it exists
                urls = self._get_selecturls_from_data(data)
                url = urls[0]

                # TODO: catchall for 'maintask.php' and 'about:home' in a preexisting tab.  but catches too many cases
                if url in ['about:home', 'http://coagmento.org/spring2016intent/instruments/maintask.php',
                           'http://coagmento.org/fall2015intent/instruments/maintask.php'] and not url in current_state[
                    'tabs']:
                    current_state['tabs'] += [url]

                (current_state['tabs'], current_state['tabIndex']) = self._select_tab(current_state['tabs'],
                                                                                current_state['tabIndex'], url)
                pageupdate_action = 'pq:select'


            elif curr_action.count('+') == curr_action.count('page') + curr_action.count(
                    'query') + 1 and curr_action.count('tabAdded') == curr_action.count('tabSelected') == 1:
                # Cases, 1 page/query, 2 page/queries, 3 page/queries
                urls = self._get_selecturls_from_data(data)
                # TODO: Just about:newtab?  Or about:newtab and about:blank?
                if not 'about:newtab' in urls:
                    print("ASSERT URLS", urls)
                assert 'about:newtab' or 'about:blank' in urls
                if 'about:newtab' in urls:
                    current_state['tabs'] += ['about:newtab']
                else:
                    current_state['tabs'] += ['about:blank']
                current_state['tabIndex'] = len(current_state['tabs']) - 1
                pageupdate_action = 'pq:addselect'

            elif curr_action.count('+') == curr_action.count('page') + curr_action.count('query') and curr_action.count(
                    'tabAdded') == 1:
                # tabAdded and any number of pq actions
                # Case 1: pq count = 1: Tab added with right click "new tab".  Add a blank tab
                # Case 2: pq count = 2: Tab added with right click "new tab".  Add a blank tab
                # Case 3: pq count = 3: Tab added with right click "new tab".  Add a blank tab
                current_state['tabs'] += ['']
                pageupdate_action = 'pq:noaction'


            elif curr_action.count('+') == curr_action.count('page') + curr_action.count(
                    'query') + 1 and curr_action.count('tabClosed') == curr_action.count('tabSelected') == 1:
                # elif curr_action.count('+')==curr_action.count('page')+curr_action.count('query')+1 and curr_action.count('tabClosed')==curr_action.count('tabSelected')==1:
                # Case pq = 1: Close current tab.  Remove it from the list of url's.  page grouped with tabSelect is the one
                # Case pq = 2: Close current tab.  Remove it from list of url's
                # Case pq = 3: Close current tab. Remove it from list of url's in data.  Remaining ones are possible url's.  (probably not a "viewMyStuff" url)
                # TODO: Is the closeTab action always grouped with the tab that is closed?  In code the current tab is printed.  Since tabClosed and tabSelected are so close , we can assume they closed current tab and select was done automatically
                old_tabs = copy.copy(current_state['tabs'])
                current_state['tabs'] = self._close_tab(current_state['tabs'], current_state['tabIndex'])
                urls = self._get_selecturls_from_data(data)
                url = urls[0]

                (current_state['tabs'], current_state['tabIndex']) = self._select_tab(current_state['tabs'],
                                                                                current_state['tabIndex'], url)
                pageupdate_action = 'pq:closeselect'




            elif curr_action.count('+') == curr_action.count('page') + curr_action.count('query') and curr_action.count(
                    'tabSelected') == 1:
                # Case pq=2: Selected the page grouped with tabSelected action
                # Case pq=3: Selected the page grouped with tabSelected action
                urls = self._get_selecturls_from_data(data)
                url = urls[0]
                (current_state['tabs'], current_state['tabIndex']) = self._select_tab(current_state['tabs'],
                                                                                current_state['tabIndex'], url)
                pageupdate_action = 'pq:select'


            elif curr_action.count('+') == 1 and curr_action.count('tabClosed') == 1 and (
                    curr_action.count('page') + curr_action.count('query') == 1):
                # Tab closed.  Cannot do anything
                # Only occurs 2 times in the data
                pageupdate_action = 'pq:noaction'
            elif curr_action == 'query+tabSelected+tabSelected':
                # TODO: Just treat like a query+tabSelected: select the correct tab
                # User restored tabs and an old tab was selected
                # Occurs 1 time in the data
                urls = self._get_selecturls_from_data(data)
                url = urls[0]
                # TODO: tab select
                (current_state['tabs'], current_state['tabIndex']) = self._select_tab(current_state['tabs'],
                                                                                current_state['tabIndex'], url)
                pageupdate_action = 'pq:select'




            elif curr_action == 'page+page+page+page+page+tabAdded+tabAdded+tabClosed+tabSelected+tabSelected+tabSelected':
                # Opened bookmark on "arctic methane" tab.  Opened window that shifted to "about:blank->Cost of Arctic".  Cost of arctic dragged to main window
                # Occurs in order:
                # 1) Open bookmark action added tab while on "arctic methane"
                # 2) tabSelect to about:blank. TODO: note to self, if selected tab doesn't exist, create it?  Shouldn't normally happen
                # 3) page transition to about:blank->cost of arctic
                # 4) ignore tabClosed.  Cost of arctic dragged to main window
                # 5) Make sure cost of arctic is selected.  url: https://www.rsm.nl/about-rsm/news/detail/2992-cost-of-arctic-methane-release-could-be-size-of-global-economy-warn-experts/
                # Occurs 1 time in the data

                url = 'https://www.rsm.nl/about-rsm/news/detail/2992-cost-of-arctic-methane-release-could-be-size-of-global-economy-warn-experts/'
                current_state['tabs'] += [url]
                current_state['tabIndex'] = current_state['tabs'].index(url)
                pageupdate_action = 'pq:addselect'
            else:
                if self.curr_prev_action.get(curr_action, None) is None:
                    self.curr_prev_action[curr_action] = 1
                else:
                    self.curr_prev_action[curr_action] += 1
            self._update_pagedata(current_state, previous_state, total_state, pageupdate_action)
        except SystemExit:
            exit()
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            current_state['pq_error'] = "%s, line %d" % (exc_type.__name__, exc_tb.tb_lineno)

    def _get_urls_ids_from_frame(self,data):
        ids = data['value'].tolist()
        actions = data['action'].tolist()
        new_frame = []
        for (n, row) in data.iterrows():
            url = None
            if row['action'] == 'page':
                url = self.pageid_to_url[row['value']]
            elif row['action'] == 'query':
                url = self.queryid_to_url[row['value']]
            new_frame += [
                {'actionID': row['actionID'], 'pagequeryID': row['value'], 'url': url, 'action': row['action']}]
        return pd.DataFrame(new_frame)

    def _get_urls_from_frame(self,data):
        ids = data['value'].tolist()
        actions = data['action'].tolist()
        urls = []
        for (i, act) in zip(ids, actions):
            if act == 'page':
                urls += [self.pageid_to_url[i]]
            elif act == 'query':
                urls += [self.queryid_to_url[i]]
        return urls

    # def _init_new_query_segment(self,data, current_state, previous_state, total_state, queryID):
    #     for intention in self.intention_columns:
    #         if current_state.get(intention, None) == None:
    #             # first query
    #             current_state['previntent_' + intention] = 0
    #         else:
    #             current_state['previntent_' + intention] = current_state[intention]
    #
    #         current_state[intention] = self.queryid_to_intents[queryID][intention]
    #
    #     reformulationType = self.queryid_to_intents[queryID].get('reformulationType',None)
    #
    #     for refType in self.reformulation_types:
    #         if current_state.get('reformulationType_' + refType, None) == None:
    #             # first query
    #             current_state['prevreformulationType_' + refType] = 0
    #         else:
    #             current_state['prevreformulationType_' + refType] = current_state['reformulationType_' + refType]
    #         current_state['reformulationType_' + refType] = refType == reformulationType
    #
    #     current_state['queryID'] = queryID
    #     current_state['query'] = self.query_data[queryID]['query']
    #     total_state['query_lengths'] += [len(current_state['query'].split())]
    #     current_state['num_clicks'] = 0
    #
    #     current_state['bookmarks'] = []
    #     current_state['bookmarks_unsaved'] = []
    #     current_state['bookmark_saved_prevsegment'] = current_state['bookmark_saved_currentsegment']
    #     current_state['bookmark_saved_currentsegment'] = int(False)
    #     current_state['bookmark_unsaved_prevsegment'] = current_state['bookmark_unsaved_currentsegment']
    #     current_state['bookmark_unsaved_currentsegment'] = int(False)
    #
    #     current_state['pagedata_segment'] = dict()
    #
    #     total_state['num_queries'] += 1

    # TODO
    def _init_new_query_segment(self,data, current_state, previous_state, total_state, queryID):
        current_state['queryID'] = queryID
        current_state['query'] = self.query_data[queryID]['query']
        total_state['query_lengths'] += [len(current_state['query'].split())]
        current_state['num_clicks'] = 0
        # current_state['bookmarks'] = []
        # current_state['bookmarks_unsaved'] = []
        # current_state['bookmark_saved_prevsegment'] = current_state['bookmark_saved_currentsegment']
        # current_state['bookmark_saved_currentsegment'] = int(False)
        # current_state['bookmark_unsaved_prevsegment'] = current_state['bookmark_unsaved_currentsegment']
        # current_state['bookmark_unsaved_currentsegment'] = int(False)
        current_state['pagedata_segment'] = dict()
        total_state['num_queries'] += 1

    # TODO
    def _assert_valid_pagequery(self,data):
        assert len(set(self._get_urls_from_frame(data))) == len(data[data['action'] == 'page']) or (
        self._get_urls_from_frame(data).count(self.viewmystuff_url) > 1)

    # def _assert_valid_pagequery(self,data):
    #     assert len(set(self._get_urls_from_frame(data))) == len(data[data['action'] == 'page']) or (
    #     self._get_urls_from_frame(data).count(self.viewmystuff_url) > 1)

    def _clean_query_data(self,data):
        data_with_urls = self._get_urls_ids_from_frame(data)

        omit_actions = []
        for (n, group) in data_with_urls.groupby('url'):
            if len(group.index) > 1:
                omit_actions += [group[group['action'] == 'page']['actionID'].tolist()[0]]

        new_data = data[~data['actionID'].isin(omit_actions)]

        assert len(data[data['action'] == 'page']) == (len(new_data[new_data['action'] == 'page'])) + (
        len(new_data[new_data['action'] == 'query'])) or (self._get_urls_from_frame(data).count(self.viewmystuff_url) > 1)
        return new_data

    # TODO
    def _assign_queryid(self,data, current_state, previous_state, total_state):
        ids = []
        for d in data:
            ids += [row['value'] for (n, row) in d.iterrows() if row['action'] in ['page', 'query']]
        for val in ids:
            if self.intent_queryids.get(val, None) != None:
                self._init_new_query_segment(data, current_state, previous_state, total_state, val)
                break

    # def _assign_queryid(self,data, current_state, previous_state, total_state):
    #     ids = []
    #     for d in data:
    #         ids += [row['value'] for (n, row) in d.iterrows() if row['action'] in ['page', 'query']]
    #     for val in ids:
    #         if self.intent_queryids.get(val, None) != None:
    #             self._init_new_query_segment(data, current_state, previous_state, total_state, val)
    #             break

    # data: list of frames
    # Assumes: there is only ever 1 url grouped with a tabSelect in the data
    # Assumes: 'tabSelected+pq' action are in some frame
    def _get_selecturls_from_data(self,data):
        subdata = [d for d in data if 'tabSelected' in d['action'].tolist()]
        subdata = subdata[0]
        vals = [v for d in data for v in d['value'].tolist()]
        urls = [self.queryid_to_url[v] if v < self.queryid_threshold else self.pageid_to_url[v]]
        if len(set(urls)) != 1:
            print("SET URLS", urls)
            exit()

        return urls

    # Remove current tab from tabs
    # current_tabs: list of urls
    # tabIndex: tab to remove
    # Assumes: tabIndex is within defined range
    def _close_tab(self,current_tabs, tabIndex):
        current_tabs = current_tabs[0:tabIndex] + current_tabs[tabIndex + 1:]
        return current_tabs

    # current_tabs: the current tabs
    # url: the url to select
    # Case 1) url exists
    # Case 2) url exists in a different form (i.e. remove # from google search string
    # Case 3)
    def _select_tab(self,current_tabs, currentTabIndex, url):
        try:
            try:
                tabIndex = current_tabs.index(url)
                return (current_tabs, tabIndex)
            except ValueError:

                is_shortened_url = False
                tabs_alt = [u[0:u.rfind('#')] if (u.startswith('https://www.google.com/search') and '#' in u) else u for
                            u in current_tabs]
                url_alt = url[0:url.rfind('#')] if url.startswith(
                    'https://www.google.com/search') and '#' in url else url

                if url.startswith('https://www.google.com/search') and '#' in url:
                    is_shortened_url = url_alt in tabs_alt

                if is_shortened_url:
                    if currentTabIndex < len(tabs_alt) and tabs_alt[currentTabIndex] == url_alt:
                        current_tabs[currentTabIndex] = url
                        return (current_tabs, currentTabIndex)
                    else:
                        tabIndex = tabs_alt.index(url_alt)
                        current_tabs[tabIndex] = url
                        return (current_tabs, tabIndex)
                else:
                    tabIndex = current_tabs.index('')
                    current_tabs[tabIndex] = url
                    return (current_tabs, tabIndex)

        except TypeError:
            print(current_tabs)
            print(url)
            print("TYPE ERROR")
            exit()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()

            print(current_tabs)
            print(url)
            print("OTHER ERROR", type(e).__name__, exc_tb.tb_lineno)

    def _create_feature_vector(self,current_state, previous_state, total_state):
        f_vec = dict()
        f_vec['userID'] = total_state['userID']
        f_vec['queryID'] = current_state['queryID']

        f_vec['action_type'] = current_state['action_type']

        for survey_input in ['topic_familiarity', 'assignment_experience', 'perceived_difficulty', 'posttask_success',
                             'posttask_difficulty', 'posttask_time', 'posttask_comprehension']:
            f_vec[survey_input] = total_state[survey_input]

        # for debug purposes
        # print: 1) action type 2) tabs 3) current tab
        f_vec['pq_current_action_type'] = current_state.get('pq_current_action_type', '')
        f_vec['pq_current_action_count'] = current_state.get('pq_current_action_count', 0)
        f_vec['pq_error'] = current_state.get('pq_error', '')
        f_vec['tabs'] = str([t for t in current_state['tabs']])
        # f_vec['tabs'] = str([t[0:20] for t in current_state['tabs']])
        f_vec['number_tabs'] = len(current_state['tabs'])
        if current_state['tabIndex'] is not None and current_state['tabIndex'] < len(current_state['tabs']):
            f_vec['current_tab'] = current_state['tabs'][current_state['tabIndex']]
        else:
            f_vec['current_tab'] = "ERROR"

        f_vec['questionID'] = total_state['questionID']
        f_vec['task_topic'] = total_state['task_topic']
        f_vec['task_type'] = total_state['task_type']
        f_vec['facet_product'] = total_state['facet_product']
        f_vec['facet_level'] = total_state['facet_level']
        f_vec['facet_goal'] = total_state['facet_goal']
        f_vec['facet_named'] = total_state['facet_named']
        f_vec['search_years'] = total_state['search_years']
        f_vec['search_expertise'] = total_state['search_expertise']
        f_vec['search_frequency'] = total_state['search_frequency']
        f_vec['search_journalism'] = total_state['search_journalism']
        f_vec['post_successful'] = total_state['post_successful']
        f_vec['post_difficult'] = total_state['post_difficult']
        f_vec['post_rushed'] = total_state['post_rushed']
        f_vec['post_comprehend'] = total_state['post_comprehend']


        f_vec['localTimestamp'] = total_state['localTimestamp']

        f_vec['clicks_num_session'] = total_state['num_clicks']
        f_vec['clicks_num_segment'] = current_state['num_clicks']

        f_vec['time_total'] = total_state['total_time']
        f_vec['time_percent_relative'] = total_state['percent_time_relative']
        f_vec['time_percent_20mins'] = total_state['percent_time_20mins']

        f_vec['bookmarks_num_total'] = len(total_state['bookmarks'])
        f_vec['bookmarks_num_segment'] = len(current_state['bookmarks'])
        f_vec['bookmark_saved_currentsegment'] = current_state['bookmark_saved_currentsegment']
        f_vec['bookmark_saved_prevsegment'] = current_state['bookmark_saved_prevsegment']

        f_vec['query_length'] = len(current_state['query'].split())
        # f_vec['query_length_contentwords'] = len([w for w in current_state['query'].split() if not w in STOPWORDS])
        f_vec['queries_num'] = total_state['num_queries']
        f_vec['clicks_per_query'] = float(total_state['num_clicks']) / float(total_state['num_queries'])
        f_vec['bookmarks_per_query'] = len(total_state['bookmarks']) / float(total_state['num_queries'])
        f_vec['queries_percent'] = float(total_state['num_queries']) / float(total_state['num_queries_total'])
        f_vec['queries_lengths_mean'] = np.mean(total_state['query_lengths'])
        f_vec['queries_lengths_range'] = np.max(total_state['query_lengths']) - np.min(total_state['query_lengths'])

        f_vec['tabs_count'] = len(current_state['tabs'])

        # 'time_dwelltotal'
        # 'time_dwellfirst'
        # 'time_dwelltotaltosave'
        # 'time_opentotal'
        # 'time_opentotaltosave'
        # 'views_total'
        # 'views_totaltosave'

        def get_page_type(pt, segmentorsession):
            related_data = None
            related_bookmarks_lookup = None
            related_unsavedbookmarks_lookup = None
            if segmentorsession == 'segment':
                related_data = current_state['pagedata_segment']
                related_bookmarks_lookup = dict([(b, []) for b in current_state['bookmarks_unsaved']])
                related_unsavedbookmarks_lookup = dict([(b, []) for b in current_state['bookmarks_unsaved']])
            else:
                related_data = total_state['pagedata_total']
                related_bookmarks_lookup = dict([(b, []) for b in total_state['bookmarks']])
                related_unsavedbookmarks_lookup = dict([(b, []) for b in total_state['bookmarks_unsaved']])

            related_data = dict([(k, v) for (k, v) in related_data.iteritems() if k != ''])
            # returns dictionary of page information.  Keys are url's, values are times
            if pt == 'serp':
                return dict([(p, v) for (p, v) in related_data.iteritems() if 'google.com/search' in p])
            elif pt == 'saved':
                return dict([(p, v) for (p, v) in related_data.iteritems() if
                             related_bookmarks_lookup.get(p, None) is not None])
            elif pt == 'unsaved':
                return dict([(p, v) for (p, v) in related_data.iteritems() if
                             related_unsavedbookmarks_lookup.get(p, None) is not None])
            elif pt == 'notsaved':
                return dict([(p, v) for (p, v) in related_data.iteritems() if (
                related_bookmarks_lookup.get(p, None) is not None and related_unsavedbookmarks_lookup.get(p,
                                                                                                          None) is not None)])
            elif pt == 'content':
                return dict([(p, v) for (p, v) in related_data.iteritems() if 'google.com/search' not in p])

        for seg_session in ['segment', 'session']:
            for pagetype in ['serp', 'saved', 'unsaved', 'notsaved', 'content']:
                related_pages = get_page_type(pagetype, seg_session)
                # 'time_dwelltotal'
                # 'time_dwelltotaltosave'
                # 'time_opentotal'
                # 'time_opentotaltosave'
                # 'views_total'
                # 'views_totaltosave'
                # TODO: ,'views_total','views_totaltosave', 'time_dwellfirst'
                for valuetype in ['time_dwelltotal', 'time_dwelltotaltosave', 'time_opentotal', 'time_opentotaltosave']:
                    values_list = [values[valuetype] for (p, values) in related_pages.iteritems()]
                    if len(values_list) == 0:
                        f_vec["%s_mean_%s_%s" % (valuetype, pagetype, seg_session)] = 0.0
                        f_vec["%s_total_%s_%s" % (valuetype, pagetype, seg_session)] = 0.0
                    else:
                        f_vec["%s_mean_%s_%s" % (valuetype, pagetype, seg_session)] = np.mean(
                            [values[valuetype] for (p, values) in related_pages.iteritems()])
                        f_vec["%s_total_%s_%s" % (valuetype, pagetype, seg_session)] = np.sum(
                            [values[valuetype] for (p, values) in related_pages.iteritems()])
                    # print("SAVING:", "%s_mean_%s_%s" % (valuetype, pagetype, seg_session))

        f_vec['pages_num_segment'] = len(set([url for url in current_state['pagedata_segment'].keys() if
                                              url not in ['about:blank', 'about:newtab', '']]))
        f_vec['pages_num_session'] = len(set(
            [url for url in total_state['pagedata_total'].keys() if url not in ['about:blank', 'about:newtab', '']]))
        f_vec['pages_per_query'] = f_vec['pages_num_session'] / float(total_state['num_queries'])
        f_vec['sources_num_segment'] = len(set(
            [urlparse(url).netloc for url in current_state['pagedata_segment'].keys() if
             url not in ['about:blank', 'about:newtab', '']]))

        # f_vec['total_searchsources'] =


        # TODO: total time elapsed?

        # TODO!!!!


        f_vec['intenttotal_current_id'] = abs(current_state['id_start'])+abs(current_state['id_more'])
        f_vec['intenttotal_current_learn'] = abs(current_state['learn_domain'])+abs(current_state['learn_database'])
        f_vec['intenttotal_current_findaccessobtain'] = abs(current_state['find_known'])+abs(current_state['find_specific'])+abs(current_state['find_common'])+ \
                                                            abs(current_state['find_without'])+abs(current_state['access_item'])+ \
                                                                abs(current_state['access_common'])+abs(current_state['access_area'])+ \
                                                                    abs(current_state['obtain_specific']) + abs(current_state['obtain_part'])+abs(current_state['obtain_whole'])
        f_vec['intenttotal_current_keep'] = abs(current_state['keep_link'])
        f_vec['intenttotal_current_evaluate'] = abs(current_state['evaluate_correctness'])+abs(current_state['evaluate_specificity'])+abs(current_state['evaluate_usefulness'])+abs(current_state['evaluate_best'])+abs(current_state['evaluate_duplication'])

        for intention in self.intention_columns:
            f_vec['intent_current_' + intention] = current_state[intention]
            f_vec['intent_prev_' + intention] = current_state['previntent_' + intention]

        for refType in self.reformulation_types:
            f_vec['reformulationType_current_' + refType] = int(current_state['reformulationType_' + refType])
            f_vec['reformulationType_prev_' + refType] = int(current_state['prevreformulationType_' + refType])

        return f_vec

    def _update_total_state(self,current_state, previous_state, total_state):
        # Metadata should be constant
        total_state['localTimestamp'] = current_state['localTimestamp']
        total_state['total_time'] = total_state['localTimestamp'] - total_state['startTimestamp']
        total_state['percent_time_relative'] = float(total_state['total_time']) / float(
            total_state['endTimestamp'] - total_state['startTimestamp'])
        total_state['percent_time_20mins'] = float(total_state['total_time']) / 1200000.0

    def _init_states(self,current_state, total_state, previous_state):

        current_state['queryID'] = None
        current_state['num_clicks'] = 0
        previous_state['num_clicks'] = 0
        total_state['num_clicks'] = 0

        total_state['num_queries'] = 0
        total_state['query_lengths'] = []

        current_state['bookmarks'] = []
        previous_state['bookmarks'] = []
        total_state['bookmarks'] = []
        total_state['bookmarks_unsaved'] = []
        current_state['bookmarks_unsaved'] = []

        current_state['bookmark_saved_currentsegment'] = int(False)
        previous_state['bookmark_saved_currentsegment'] = int(False)

        current_state['bookmark_saved_prevsegment'] = int(False)
        previous_state['bookmark_saved_prevsegment'] = int(False)

        current_state['bookmark_unsaved_currentsegment'] = int(False)
        previous_state['bookmark_unsaved_currentsegment'] = int(False)

        current_state['bookmark_unsaved_prevsegment'] = int(False)
        previous_state['bookmark_unsaved_prevsegment'] = int(False)

        # current_state['query'] = None
        # previous_state['query'] = None
        #
        # current_state['localTimestamp'] = None
        # previous_state['localTimestamp'] = None
        current_state['tabs'] = []
        current_state['pq_current_action_count'] = 0
        current_state['tabIndex'] = None
        previous_state['tab'] = []
        previous_state['tabIndex'] = None

        total_state['pagedata_total'] = dict()
        current_state['pagedata_segment'] = dict()

    def _init_states_pages(self, current_state, total_state, previous_state):

        current_state['queryID'] = None
        # current_state['num_clicks'] = 0
        # previous_state['num_clicks'] = 0
        # total_state['num_clicks'] = 0

        total_state['num_queries'] = 0
        total_state['query_lengths'] = []

        total_state['pagedata_total'] = dict()
        current_state['pagedata_segment'] = dict()

        # current_state['bookmarks'] = []
        # previous_state['bookmarks'] = []
        # total_state['bookmarks'] = []
        # total_state['bookmarks_unsaved'] = []
        # current_state['bookmarks_unsaved'] = []

        # current_state['bookmark_saved_currentsegment'] = int(False)
        # previous_state['bookmark_saved_currentsegment'] = int(False)

        # current_state['bookmark_saved_prevsegment'] = int(False)
        # previous_state['bookmark_saved_prevsegment'] = int(False)

        # current_state['bookmark_unsaved_currentsegment'] = int(False)
        # previous_state['bookmark_unsaved_currentsegment'] = int(False)

        # current_state['bookmark_unsaved_prevsegment'] = int(False)
        # previous_state['bookmark_unsaved_prevsegment'] = int(False)

        # current_state['tabs'] = []
        # current_state['pq_current_action_count'] = 0
        # current_state['tabIndex'] = None
        # previous_state['tab'] = []
        # previous_state['tabIndex'] = None


    def slice_frame(self,override=False):
        if self.auxiliary_datasets is not None and not override:
            print("Auxiliary datasets already created!")
            return self.auxiliary_datasets
        self.auxiliary_datasets = dict()
        dataset_byquery = []
        for (n,group) in self.full_data_frame.groupby(['queryID']):
            dataset_byquery += [group.tail(1)]
            print(group.tail(1))

        self.auxiliary_datasets['byquery'] = pd.concat(dataset_byquery)
        self.auxiliary_datasets['byquery']['facet_goal_val_specific'] = (self.auxiliary_datasets['byquery']['facet_goal']=='Specific')*1
        self.auxiliary_datasets['byquery']['topic_globalwarming'] = (self.auxiliary_datasets['byquery'][
                                                                             'task_topic'] != 'Coelacanths') * 1
        self.auxiliary_datasets['byquery']['facet_product_val_factual'] = (self.auxiliary_datasets['byquery'][
                                                                             'facet_product'] == 'Factual') * 1
        return self.auxiliary_datasets





if __name__=='__main__':
    logger = Sum14LogParser(output_csv='/Users/Matt/Desktop/features_summer2014_alldata.csv')
    logger.process_log()
    # subframes = logger.slice_frame()
    #
    # print(subframes['byquery'])
    # subframes['byquery'].to_csv('/Users/Matt/Desktop/features_nsf_byquery.csv')
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
#   ('delete_bookmarks',1440704848)
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
# current_bookmarks:[ #current bookmarks - update on save and delete
#  (bookmarkID1=int,url1=int),
#  (643,'https://en.wikipedia.org'),
#  ...
# ]
#
# queryID:int, #from video_intent_assignments
# query_data:dict,
# }
#