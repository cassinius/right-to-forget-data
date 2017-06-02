import psycopg2
import copy
import json

import datetime
DATE_FORMAT = '%Y%m%d%H%M%S'

DB_NAME = 'iml_anonymization'
DB_HOST = '127.0.0.1'
DB_USER = 'iml_admin'
DB_PASS = 'iml_pass'
DB_TABLE_RAW = 'iml_requests_raw'
DB_TABLE_RESULTS = 'iml_results'

conn = None
cur = None

try:
    conn = psycopg2.connect("dbname=" + DB_NAME + " user=" + DB_USER + " host=" + DB_HOST + " password=" + DB_PASS)
    cur = conn.cursor()
except:
    print "I am unable to connect to the database"


# Just a test
# print conn
# cur.execute("SELECT * FROM " + DB_TABLE_RAW)
# rows = cur.fetchall()
# 
# print "\nShow me the raw requests:\n"
# for row in rows:
#     print "   ", row
#
# timestamp = datetime.datetime.now().strftime(DATE_FORMAT)
# db_json = {
#     "bla"   : "hoo",
#     "1"     : 2,
#     "arr"   : [1, 2, 3]
# }
#
# try:
#     query = """INSERT INTO %s (timestamp, request_raw) VALUES (%s, '%s')""" % (DB_TABLE_RAW, timestamp, json.dumps(db_json))
#     print query
#     cur.execute(query)
#     conn.commit()
# except:
#     print 'blaaa'



def storeRawRequest(request_json, timestamp):
    db_json = copy.deepcopy(request_json)
    del db_json["csv"]

    # print "Storing raw request:"
    # print timestamp
    # print db_json

    try:
        query = """INSERT INTO %s (timestamp, request_raw) VALUES (%s, '%s')""" % (DB_TABLE_RAW, timestamp, json.dumps(db_json))
        cur.execute(query)
        conn.commit()
    except:
        print "DB transaction failed... RAW request was not saved!"



def storeResult(request, overall_result):
    print "Storing result set..."
